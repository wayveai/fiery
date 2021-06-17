from typing import Optional

import torch
from pytorch_lightning.metrics.metric import Metric
from pytorch_lightning.metrics.functional.classification import stat_scores_multiple_classes
from pytorch_lightning.metrics.functional.reduction import reduce


MATCH_THRESHOLD = 0.5


class IntersectionOverUnion(Metric):
    """Computes intersection-over-union."""
    def __init__(
        self,
        n_classes: int,
        ignore_index: Optional[int] = None,
        absent_score: float = 0.0,
        reduction: str = 'none',
        compute_on_step: bool = False,
    ):
        super().__init__(compute_on_step=compute_on_step)

        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.absent_score = absent_score
        self.reduction = reduction

        self.add_state('true_positive', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('false_positive', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('false_negative', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('support', default=torch.zeros(n_classes), dist_reduce_fx='sum')

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        tps, fps, _, fns, sups = stat_scores_multiple_classes(prediction, target, self.n_classes)

        self.true_positive += tps
        self.false_positive += fps
        self.false_negative += fns
        self.support += sups

    def compute(self):
        scores = torch.zeros(self.n_classes, device=self.true_positive.device, dtype=torch.float32)

        for class_idx in range(self.n_classes):
            if class_idx == self.ignore_index:
                continue

            tp = self.true_positive[class_idx]
            fp = self.false_positive[class_idx]
            fn = self.false_negative[class_idx]
            sup = self.support[class_idx]

            # If this class is absent in the target (no support) AND absent in the pred (no true or false
            # positives), then use the absent_score for this class.
            if sup + tp + fp == 0:
                scores[class_idx] = self.absent_score
                continue

            denominator = tp + fp + fn
            score = tp.to(torch.float) / denominator
            scores[class_idx] = score

        # Remove the ignored class index from the scores.
        if (self.ignore_index is not None) and (0 <= self.ignore_index < self.n_classes):
            scores = torch.cat([scores[:self.ignore_index], scores[self.ignore_index+1:]])

        return reduce(scores, reduction=self.reduction)


class PanopticMetric(Metric):
    def __init__(
        self,
        n_classes: int,
        temporally_consistent: bool = True,
        vehicles_id: int = 1,
        compute_on_step: bool = False,
        include_trajectory_metrics: bool = False,
        pixel_resolution: float = 1.0,
    ):
        super().__init__(compute_on_step=compute_on_step)

        self.n_classes = n_classes
        self.temporally_consistent = temporally_consistent
        self.vehicles_id = vehicles_id
        self.include_trajectory_metrics = include_trajectory_metrics
        self.pixel_resolution = pixel_resolution
        self.keys = ['iou', 'true_positive', 'false_positive', 'false_negative']

        self.add_state('iou', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('true_positive', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('false_positive', default=torch.zeros(n_classes), dist_reduce_fx='sum')
        self.add_state('false_negative', default=torch.zeros(n_classes), dist_reduce_fx='sum')

        if self.include_trajectory_metrics:
            # Average Displacement Error
            self.add_state('ade', default=torch.zeros(n_classes), dist_reduce_fx='sum')
            # Final Displacement Error
            self.add_state('fde', default=torch.zeros(n_classes), dist_reduce_fx='sum')
            self.add_state('fde_counts', default=torch.zeros(n_classes), dist_reduce_fx='sum')

    def update(self, pred_instance, gt_instance):
        """
        Update state with predictions and targets.

        Parameters
        ----------
            pred_instance: (b, s, h, w)
                Temporally consistent instance segmentation prediction.
            gt_instance: (b, s, h, w)
                Ground truth instance segmentation.
        """
        batch_size, sequence_length = gt_instance.shape[:2]
        # Process labels
        assert gt_instance.min() == 0, 'ID 0 of gt_instance must be background'
        pred_segmentation = (pred_instance > 0).long()
        gt_segmentation = (gt_instance > 0).long()

        for b in range(batch_size):
            unique_id_mapping = {}
            for t in range(sequence_length):
                is_end_of_trajectory = t == sequence_length - 1
                result = self.panoptic_metrics(
                    pred_segmentation[b, t].detach(),
                    pred_instance[b, t].detach(),
                    gt_segmentation[b, t],
                    gt_instance[b, t],
                    unique_id_mapping,
                    is_end_of_trajectory,
                )

                self.iou += result['iou']
                self.true_positive += result['true_positive']
                self.false_positive += result['false_positive']
                self.false_negative += result['false_negative']

                if self.include_trajectory_metrics:
                    self.ade += result['ade']

                    if is_end_of_trajectory:
                        self.fde += result['fde']
                        self.fde_counts += result['fde_counts']

    def compute(self):
        denominator = torch.maximum(
            (self.true_positive + self.false_positive / 2 + self.false_negative / 2),
            torch.ones_like(self.true_positive)
        )
        pq = self.iou / denominator
        sq = self.iou / torch.maximum(self.true_positive, torch.ones_like(self.true_positive))
        rq = self.true_positive / denominator

        output_metrics = {'pq': pq,
                          'sq': sq,
                          'rq': rq,
                           # If 0, it means there wasn't any detection.
                          'denominator': (self.true_positive + self.false_positive / 2 + self.false_negative / 2),
                          }
        if self.include_trajectory_metrics:
            ade = self.ade / torch.maximum(self.true_positive, torch.ones_like(self.true_positive))
            output_metrics['ade'] = self.pixel_resolution * ade

            fde = self.fde / torch.maximum(self.fde_counts, torch.ones_like(self.fde_counts))
            output_metrics['fde'] = self.pixel_resolution * fde

        return output_metrics

    def panoptic_metrics(self, pred_segmentation, pred_instance, gt_segmentation, gt_instance, unique_id_mapping,
                         is_end_of_trajectory=False):
        """
        Computes panoptic quality metric components.

        Parameters
        ----------
            pred_segmentation: [H, W] range {0, ..., n_classes-1} (>= n_classes is void)
            pred_instance: [H, W] range {0, ..., n_instances} (zero means background)
            gt_segmentation: [H, W] range {0, ..., n_classes-1} (>= n_classes is void)
            gt_instance: [H, W] range {0, ..., n_instances} (zero means background)
            unique_id_mapping: instance id mapping to check consistency
        """
        n_classes = self.n_classes

        result = {key: torch.zeros(n_classes, dtype=torch.float32, device=gt_instance.device) for key in self.keys}

        if self.include_trajectory_metrics:
            result['ade'] = torch.zeros(n_classes, dtype=torch.float32, device=gt_instance.device)
            if is_end_of_trajectory:
                result['fde'] = torch.zeros(n_classes, dtype=torch.float32, device=gt_instance.device)
                result['fde_counts'] = torch.zeros(n_classes, dtype=torch.float32, device=gt_instance.device)
            h, w = gt_instance.shape
            grid = torch.stack(torch.meshgrid(
                torch.arange(h, dtype=torch.float, device=gt_instance.device),
                torch.arange(w, dtype=torch.float, device=gt_instance.device)
            )).view(2, -1)

        assert pred_segmentation.dim() == 2
        assert pred_segmentation.shape == pred_instance.shape == gt_segmentation.shape == gt_instance.shape

        n_instances = int(torch.cat([pred_instance, gt_instance]).max().item())
        n_all_things = n_instances + n_classes  # Classes + instances.
        n_things_and_void = n_all_things + 1

        # Now 1 is background; 0 is void (not used). 2 is vehicle semantic class but since it overlaps with
        # instances, it is not present.
        # and the rest are instance ids starting from 3
        prediction, pred_to_cls = self.combine_mask(pred_segmentation, pred_instance, n_classes, n_all_things)
        target, target_to_cls = self.combine_mask(gt_segmentation, gt_instance, n_classes, n_all_things)

        # Compute ious between all stuff and things
        # hack for bincounting 2 arrays together
        x = prediction + n_things_and_void * target
        bincount_2d = torch.bincount(x.long(), minlength=n_things_and_void ** 2)
        if bincount_2d.shape[0] != n_things_and_void ** 2:
            raise ValueError('Incorrect bincount size.')
        conf = bincount_2d.reshape((n_things_and_void, n_things_and_void))
        # Drop void class
        conf = conf[1:, 1:]

        # Confusion matrix contains intersections between all combinations of classes
        union = conf.sum(0).unsqueeze(0) + conf.sum(1).unsqueeze(1) - conf
        iou = torch.where(union > 0, (conf.float() + 1e-9) / (union.float() + 1e-9), torch.zeros_like(union).float())

        # In the iou matrix, first dimension is target idx, second dimension is pred idx.
        # Mapping will contain a tuple that maps prediction idx to target idx for segments matched by iou.
        mapping = (iou > MATCH_THRESHOLD).nonzero(as_tuple=False)

        # Check that classes match.
        is_matching = pred_to_cls[mapping[:, 1]] == target_to_cls[mapping[:, 0]]
        mapping = mapping[is_matching]
        tp_mask = torch.zeros_like(conf, dtype=torch.bool)
        tp_mask[mapping[:, 0], mapping[:, 1]] = True

        # First ids correspond to "stuff" i.e. semantic seg.
        # Instance ids are offset accordingly
        for target_id, pred_id in mapping:
            cls_id = pred_to_cls[pred_id]

            if self.temporally_consistent and cls_id == self.vehicles_id:
                if target_id.item() in unique_id_mapping and unique_id_mapping[target_id.item()] != pred_id.item():
                    # Not temporally consistent
                    result['false_negative'][target_to_cls[target_id]] += 1
                    result['false_positive'][pred_to_cls[pred_id]] += 1
                    unique_id_mapping[target_id.item()] = pred_id.item()
                    continue

            result['true_positive'][cls_id] += 1
            result['iou'][cls_id] += iou[target_id][pred_id]
            unique_id_mapping[target_id.item()] = pred_id.item()

            if self.temporally_consistent and self.include_trajectory_metrics:
                if cls_id == self.vehicles_id:
                    # Add 1 to class because void class was dropped
                    l2_distance = self.compute_l2_distance_centers(
                        grid, target, prediction, target_id + 1, pred_id + 1
                    )
                    result['ade'][cls_id] += l2_distance

                    if is_end_of_trajectory:
                        result['fde'][cls_id] += l2_distance
                        result['fde_counts'][cls_id] += 1

        for target_id in range(n_classes, n_all_things):
            # If this is a true positive do nothing.
            if tp_mask[target_id, n_classes:].any():
                continue
            # If this target instance didn't match with any predictions and was present set it as false negative.
            if target_to_cls[target_id] != -1:
                result['false_negative'][target_to_cls[target_id]] += 1

        for pred_id in range(n_classes, n_all_things):
            # If this is a true positive do nothing.
            if tp_mask[n_classes:, pred_id].any():
                continue
            # If this predicted instance didn't match with any prediction, set that predictions as false positive.
            if pred_to_cls[pred_id] != -1 and (conf[:, pred_id] > 0).any():
                result['false_positive'][pred_to_cls[pred_id]] += 1

        return result

    def compute_l2_distance_centers(self, grid, target, prediction, target_id, pred_id):
        target_center = grid[:, target == target_id].mean(dim=-1)
        pred_center = grid[:, prediction == pred_id].mean(dim=-1)
        return torch.linalg.norm(target_center - pred_center, ord=2)

    def combine_mask(self, segmentation: torch.Tensor, instance: torch.Tensor, n_classes: int, n_all_things: int):
        """Shifts all things ids by num_classes and combines things and stuff into a single mask

        Returns a combined mask + a mapping from id to segmentation class.
        """
        instance = instance.view(-1)
        instance_mask = instance > 0
        instance = instance - 1 + n_classes

        segmentation = segmentation.clone().view(-1)
        segmentation_mask = segmentation < n_classes  # Remove void pixels.

        # Build an index from instance id to class id.
        instance_id_to_class_tuples = torch.cat(
            (
                instance[instance_mask & segmentation_mask].unsqueeze(1),
                segmentation[instance_mask & segmentation_mask].unsqueeze(1),
            ),
            dim=1,
        )
        instance_id_to_class = -instance_id_to_class_tuples.new_ones((n_all_things,))
        instance_id_to_class[instance_id_to_class_tuples[:, 0]] = instance_id_to_class_tuples[:, 1]
        instance_id_to_class[torch.arange(n_classes, device=segmentation.device)] = torch.arange(
            n_classes, device=segmentation.device
        )

        segmentation[instance_mask] = instance[instance_mask]
        segmentation += 1  # Shift all legit classes by 1.
        segmentation[~segmentation_mask] = 0  # Shift void class to zero.

        return segmentation, instance_id_to_class


class NSamplesPanopticMetric(Metric):
    """
    Computes IOU.
    Args:
        ...
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When `None`, DDP
            will be used to perform the allgather. default: None
    """

    def __init__(
        self,
        num_classes: int,
        temporally_consistent: bool = True,
        include_trajectory_metrics: bool = False,
        pixel_resolution: float = 1.0,
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group=None,
        # dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            # dist_sync_fn=dist_sync_fn,
        )

        self.num_classes = num_classes

        self.panoptic_metric = PanopticMetric(num_classes, temporally_consistent=temporally_consistent,
                                              include_trajectory_metrics=include_trajectory_metrics,
                                              pixel_resolution=pixel_resolution,
                                              compute_on_step=compute_on_step,
                                              )

        self.add_state("ade", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fde", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("n_elements", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, pred_instance_samples: torch.Tensor, gt_instance: torch.Tensor):
        """
        Update state with predictions and targets.
        Args:
            pred_instance_samples: (b, n_samples, s, h, w)
            gt_instance: (b, s, h, w)
        """
        assert len(pred_instance_samples.shape) == 5
        batch_size, n_samples = pred_instance_samples.shape[:2]

        for b in range(batch_size):
            ade_list = []
            fde_list = []
            for i in range(n_samples):
                self.panoptic_metric(pred_instance_samples[b:b+1, i], gt_instance[b:b+1])
                score = self.panoptic_metric.compute()
                self.panoptic_metric.reset()
                if score['denominator'][1] > 0:
                    ade_list.append(score['ade'][1])
                    fde_list.append(score['fde'][1])

            if len(ade_list) > 0:
                ade_list = torch.stack(ade_list, dim=-1)
                fde_list = torch.stack(fde_list, dim=-1)

                index = torch.argmin(ade_list, dim=-1)

                self.ade += ade_list[index]
                self.fde += fde_list[index]
                self.n_elements += 1

    def compute(self):
        ade = self.ade / self.n_elements
        fde = self.fde / self.n_elements

        return {'ade': ade,
                'fde': fde,
                }
