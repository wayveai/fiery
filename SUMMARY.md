## Problem formulation

FIERY addresses **probabilistic future prediction of dynamic agents in bird’s‑eye view (BEV)** from surround monocular camera streams, replacing detect–track–predict pipelines and avoiding HD maps or LiDAR.[^1]
Given past observations $O_{1:t}$ (6 RGB cameras per time step) and past ego‑motion $a_{1:t-1}$, the model predicts a sequence of BEV labels $y_{t:t+H}$ comprising per‑cell semantic class, instance centerness, instance offsets, and future flow over a fixed ground‑plane grid.[^1]

## BEV feature lifting and projection

Each camera image $I_k^t$ at time $t$ is encoded with EfficientNet‑B4 into features $e_k^t \in \mathbb{R}^{(C+D)\times H_e \times W_e}$, split into appearance features $e_{k,t,C}\in\mathbb{R}^{C\times H_e\times W_e}$ and depth logits $e_{k,t,D}\in\mathbb{R}^{D\times H_e\times W_e}$.[^1]
A tensor $u_k^t \in \mathbb{R}^{C\times D\times H_e\times W_e}$ is formed by an outer product

$$
u_k^t = e_{k,t,C} \otimes e_{k,t,D}
$$

which can be interpreted as distributing feature mass across discrete depth planes $[D_{\min}, D_{\max}]$ with 1 m spacing; these 3D features are unprojected using known intrinsics/extrinsics and vertically pooled over height into BEV feature maps $x_t\in\mathbb{R}^{C\times H\times W}$ with $H=W=200$ (0.5 m resolution over 100 m × 100 m around the ego).[^1]

## Ego‑motion warping and temporal encoder

To factor out ego‑motion, past BEV features $(x_1,\dots,x_t)$ are warped into the current reference frame using SE(3) poses $a_i$ and a Spatial Transformer $S$:[^1]

$$
\tilde{x}_t^{\,i} = S\big(x_i,\; a_{t-1}\cdot a_{t-2}\cdots a_i\big), \quad i\in\{1,\dots,t-1\}.
$$

The sequence $\{\tilde{x}_t^{\,1},\dots,\tilde{x}_t^{\,t-1}, x_t\}$ plus broadcasted action channels is fed to a 3D CNN temporal model $T$ to produce a spatio‑temporal state

$$
s_t = T(\tilde{x}_t^{\,1}, \dots, \tilde{x}_t^{\,t-1}, x_t) \in \mathbb{R}^{64\times 200\times 200}.
$$[^1]

Each temporal block combines 3D convolutions with temporal receptive field 2, 3D global average pooling, feature compression (1×1×1 conv), and residual/skip connections.[^1]

## Conditional variational latent model

FIERY models future stochasticity via two **diagonal Gaussian** posteriors over a latent vector $\eta_t\in\mathbb{R}^L$ ($L=32$):[^1]

- Present distribution $P(\eta_t\mid s_t) = \mathcal{N}(\mu_{t,\text{present}}, \sigma^2_{t,\text{present}})$
- Future distribution $F(\eta_t\mid s_t, y_{t+1:t+H}) = \mathcal{N}(\mu_{t,\text{future}}, \sigma^2_{t,\text{future}})$.[^1]

Both are implemented as CNNs over $s_t$ (and concatenated future labels for $F$), with four downsampling residual blocks and global pooling, regressing $\mu$ and $\log \sigma$.[^1]
The probabilistic objective is a mode‑covering KL term

$$
\mathcal{L}_{\text{prob}} = D_{\text{KL}}\big(F(\cdot\mid s_t,y_{t+1:t+H})\;\Vert\;P(\cdot\mid s_t)\big),
$$

weighted by $\lambda_{\text{prob}}=100$, encouraging the present prior to cover future modes while the posterior provides correct futures during training.[^1]

During training, $\eta_t\sim F$ is sampled via reparameterization to condition predictions on the observed future; at inference, samples $\eta_t\sim P$ generate diverse futures from the same past.[^1]

## Recurrent BEV future predictor and decoder

The future predictor is a convolutional GRU unrolled for $H$ steps, taking as input $(s_t,\eta_t)$ at the first step and then its own hidden state.[^1]

- Repeated block: ConvGRU → 3 residual 2D conv blocks
- Outputs future states $\hat{s}_{t+1},\dots,\hat{s}_{t+H}$.[^1]

A shared BEV decoder $D$ maps each state $\hat{s}_{t+j}$ to four heads:[^1]

- Semantic logits (vehicles vs background)
- Centerness heatmap (2D Gaussian around instance centers)
- Offset field (2‑D vector per pixel toward its instance center)
- Future flow field (2‑D displacement vector for dynamic agents between $t+j$ and $t+j+1$).

Formally, for each $j\in\{0,\dots,H\}$,

$$
\hat{y}_{t+j} = D(\hat{s}_{t+j}),
$$

with $\hat{s}_t = s_t$.[^1]

## Instance construction, tracking, and trajectories

For each predicted future frame:[^1]

- Instance centers are extracted via non‑maximum suppression on the centerness heatmap.
- Pixels with positive semantic label are assigned to nearest centers using the offset vectors, yielding instance masks in BEV.
- Flow is used for temporal identity consistency: instance centers at time $t+j$ are warped forward using predicted flow and matched to centers at $t+j+1$ via Hungarian matching, giving temporally consistent instance IDs and hence non‑parametric trajectories.[^1]

Labels are constructed by projecting 3D bounding boxes of vehicles into BEV, computing Gaussian centerness maps, offset vectors to instance centers, and per‑instance flow between consecutive frames.[^1]

## Training regime and losses

The model uses 1.0 s of past context and predicts 2.0 s into the future (3 past + 4 future frames at 2 Hz for nuScenes; 6 past + 10 future at 5 Hz for Lyft).[^1]
Training uses Adam with learning rate $3\times 10^{-4}$, batch size 12 on 4×V100 GPUs for 20 epochs, mixed precision; total parameters ≈ 8.1M.[^1]

Loss components:[^1]

- Semantic segmentation: top‑k cross‑entropy on BEV grid (k = 25% hardest pixels) to mitigate background dominance.
- Centerness: L2 loss.
- Offset and flow: L1 losses.
- Task uncertainty weighting: per‑task homoscedastic uncertainty to balance heads.
- Temporal discounting: exponential factor $\gamma=0.95$ for future steps.

Total loss is the sum of these terms plus $\lambda_{\text{prob}}\mathcal{L}_{\text{prob}}$.[^1]

## Evaluation metrics

Two main metrics are used:[^1]

- **Video Panoptic Quality (VPQ):**

$$
\text{VPQ} = 
\frac{\sum_{t=0}^{H}\sum_{(p_t,q_t)\in \text{TP}_t} \text{IoU}(p_t, q_t)}
     {\sum_{t} |\text{TP}_t| + \tfrac{1}{2}|\text{FP}_t| + \tfrac{1}{2}|\text{FN}_t|},
$$

where TP/FP/FN are defined over spatio‑temporal instances with IoU > 0.5 and consistent ID tracking.
- **Generalised Energy Distance (GED):** with distance $d(x,y)=1-\text{VPQ}(x,y)$,

$$
D_{\text{GED}} = 2\mathbb{E}[d(Y,\hat{Y})] - \mathbb{E}[d(\hat{Y},\hat{Y}')],
$$

using multiple predicted futures $\hat{Y},\hat{Y}'$; smaller is better, capturing both accuracy and diversity.[^1]


## Quantitative results

For single‑frame BEV semantic segmentation, FIERY Static (H = 0) surpasses VED, PON, VPN, STA, Lift‑Splat, and Fishing Net on nuScenes; adding temporal context (full FIERY) further improves IoU due to better handling of occlusions and partial observability.[^1]
For 2.0 s future instance prediction, FIERY significantly outperforms:

- Static (repeat current frame in future)
- Extrapolation (linear trajectory extrapolation from FIERY Static tracks)

on both nuScenes and Lyft, in both short‑range (30 × 30 m) and long‑range (100 × 100 m) VPQ and IoU.[^1]

## Ablation findings

Ablations on nuScenes (2.0 s horizon) show:[^1]

- No temporal context: VPQ close to Static, confirming the need for past frames.
- No transformation to present frame: large drop, indicating importance of ego‑motion compensation before temporal convolutions.
- No unrolling (single‑shot future prediction from $s_t$): clear degradation, showing that sequential conditioning across future steps is beneficial.
- No future flow: hurts VPQ by breaking temporal instance consistency.
- Uniform depth lifting (Orthographic Feature Transform): underperforms learned depth probabilities, validating the probabilistic depth distribution.
- Deterministic model (no variational distributions): lower VPQ and worse GED compared to FIERY, indicating that the conditional VAE improves both accuracy and multimodality.[^1]

On GED with 10 samples and 2.0 s horizon, FIERY achieves the lowest (best) scores versus M‑Head, Bayesian Dropout, and classical VAE baselines, evidencing better trade‑off between sharpness and diversity of future panoptic BEV sequences.[^1]

## Behaviour of the learned latent space and horizon

PCA plus Gaussian mixture clustering of states $s_t$ shows that clusters correspond to semantically coherent driving scenarios (approaching intersections, cruising, traffic jams, etc.), providing a state‑space explanation for how multiple futures arise from similar initial conditions under the probabilistic latent model.[^1]
VPQ as a function of horizon from 1–8 s exhibits saturation and decline beyond ≈6 s, reflecting growing uncertainty and limited observability of agents far into the future.[^1]

<div align="center">⁂</div>

[^1]: 2104.10490v3.pdf

