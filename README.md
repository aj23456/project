# project
Project Documentation and Technical Architecture: Unsupervised Anomaly Detection in Mars HiRISE Orbital Imagery
1. Executive Summary and Observational Context
The exploration of the Martian surface has been fundamentally transformed by the High-Resolution Imaging Science Experiment (HiRISE), an instrument operating aboard the National Aeronautics and Space Administration (NASA) Mars Reconnaissance Orbiter (MRO) since 2006. Engineered as a 0.5-meter aperture reflecting telescope—the largest ever deployed in a deep space mission—HiRISE captures panchromatic and multispectral imagery at a spatial resolution of roughly 0.3 meters (approximately one foot) per pixel from an orbital altitude of 200 to 400 kilometers. Operating as a pushbroom scanner that utilizes Time-Delay Integration (TDI) across up to 128 detector rows, the instrument achieves extraordinary signal-to-noise ratios, resolving surface features as diminutive as a kitchen table. This resolution is critical for evaluating future landing sites and investigating dynamic geological phenomena such as recurring slope lineae (RSL), aeolian dune migration, and fresh impact cratering.   

However, the sheer volume of high-resolution telemetry downlinked from the MRO renders manual, comprehensive inspection mathematically impossible. The data is processed into Reduced Data Records (RDRs), which are subsequently map-projected for cartographic alignment, generating a petabyte-scale archive of Martian surface topologies. Identifying statistically significant anomalies within this vast repository requires advanced computational frameworks. Because the definition of a Martian "anomaly" is completely unbounded—encompassing everything from previously unobserved geological structures and localized sensor degradation to artificial telemetry corruption or spliced frames—supervised machine learning paradigms are inherently unsuitable. There is no exhaustive taxonomy of Martian anomalies upon which a classifier could be trained.   

This project documentation details an end-to-end unsupervised machine learning pipeline designed to isolate and interpret anomalies within the HiRISE RDR map-projected archive. By synthesizing deep latent representation learning via a custom Convolutional Autoencoder (CAE), latent space whitening, an Isolation Forest novelty engine, and Extreme Value Theory (EVT) statistical thresholding, the architecture isolates morphological outliers without requiring any labeled training data. Furthermore, the pipeline establishes a framework for pixel-wise spatial interpretability, translating opaque neural network residuals into actionable geological hypotheses.   

2. System Architecture and Operational Directives
To satisfy the requirements of reproducible execution, this section delineates the operational prerequisites, environmental layout, and data ingestion constraints required to run the pipeline. This serves as the primary operational directive (or initialization procedure) for deploying the anomaly detection system on local or cloud-based infrastructure.

2.1 Repository Structure and Execution Graph
The pipeline demands a strictly defined directory hierarchy to manage the ingestion of map-projected crop data and the subsequent emission of learned artifacts, model weights, and visualization heatmaps. The foundational execution environment must be structured as outlined in the following layout.   

Directory / File Path	Functional Description and Content Expected
data/map-proj-v3/	
The primary ingestion directory containing approximately 10,000 uncompressed grayscale image crops (227×227 pixels), stripped of all categorical labels.

data/source_image_metadata.csv	
The telemetry registry detailing the parent observational context. Essential columns include source_image_id, latitude, longitude, sun_angle, season, and resolution.

data/crop_map.csv	
An optional relational mapping file utilized exclusively if the crop filenames do not inherently encode the standard HiRISE observation ID prefixes (e.g., ESP, PSP, TRA).

artifacts/	The automated output directory for localized persistence.
artifacts/ae_best.pt	
The serialized PyTorch state dictionary containing the optimal learned weights of the Convolutional Autoencoder.

artifacts/latents.npy	
The extracted 256-dimensional deterministic latent embeddings for the entire archive.

artifacts/novelty_scores.csv	
The aggregated dataset appending Isolation Forest anomaly scores and EVT threshold flags to the crop index.

artifacts/figures/	
The repository for generated t-SNE projections, threshold derivation plots, and pixel-wise residual heatmaps.

  
The execution sequence is strictly linear and must follow a defined order of operations to ensure data dependency requirements are met. Initialization begins with environment configuration, seed setting (SEED = 1337), and hardware allocation mapping (prioritizing CUDA-enabled tensor operations). Phase 1 executes the training of the deep latent compression autoencoder. Phase 2 extracts the learned features and fits the Isolation Forest novelty engine. Phase 3 subjects the flagged anomalies to pixel-wise morphological analysis. Phase 4 allows for ablation testing and architectural iteration.   

2.2 Data Ingestion and the Photometric Augmentation Constraint
The raw input data consists of single-channel (grayscale) float images normalized to a bounded range of [0,1]. To achieve dimensional compatibility with a five-stage stride-2 downsampling hierarchy, the initial 227×227 crops are dynamically resized via bicubic interpolation to 224×224 pixels. This ensures that the terminal spatial resolution lands exactly on a 7×7 grid without requiring asymmetric zero-padding, which can introduce artificial boundary artifacts in convolutional networks.   

A critical element of the training paradigm is the deliberate restriction of data augmentation. The data loader restricts augmentation strictly to spatial transformations: horizontal flips, vertical flips, and 90-degree rotations. These affine transformations are geologically and physically sound because orbital nadir imagery of a planetary surface lacks a definitive "up" orientation.   

Crucially, brightness and contrast jittering are entirely prohibited during the training phase. In standard computer vision paradigms, photometric augmentation is routinely employed to induce model invariance against varying illumination conditions. However, in the context of out-of-distribution anomaly detection, variations in photometric statistics carry primary diagnostic signals. An injected foreign frame, a sensor transmission error, or a highly unusual surface albedo (such as fresh ice in a typically dark regolith region) will inevitably differ from the surrounding terrain in its global intensity and contrast structure. Augmenting these variations away would actively instruct the encoder to become invariant to the precise anomalies the system is attempting to detect, thereby destroying the residual signal.   

3. Representation Learning: Deep Latent Compression
Phase 1 of the architecture is dedicated to modeling the normative statistical manifold of the Martian surface. The pipeline employs a custom Convolutional Autoencoder (CAE) designed and trained entirely from scratch. Pretrained weights (e.g., ImageNet priors) and external feature extractors are explicitly banned from the architecture. This constraint guarantees that the convolutional kernels specialize exclusively in the distinct spatial frequencies, aeolian textures, and impact topologies native to Mars, rather than terrestrial artifacts.   

3.1 Encoder Architecture and Latent Dimensionality
The encoder operates as a deterministic feature extractor, aggressively compressing the high-dimensional pixel space into a fixed-length bottleneck vector. The spatial reduction is achieved through five progressive blocks of stride-2 downsampling.

Architectural Stage	Spatial Dimensions	Channel Depth	Core Operations
Block 1	224×224→112×112	1 → 32	
Conv2d(stride=2), GroupNorm, SiLU, Conv2d(stride=1), GroupNorm, SiLU

Block 2	112×112→56×56	32 → 64	
Conv2d(stride=2), GroupNorm, SiLU, Conv2d(stride=1), GroupNorm, SiLU

Block 3	56×56→28×28	64 → 128	
Conv2d(stride=2), GroupNorm, SiLU, Conv2d(stride=1), GroupNorm, SiLU

Block 4	28×28→14×14	128 → 256	
Conv2d(stride=2), GroupNorm, SiLU, Conv2d(stride=1), GroupNorm, SiLU

Block 5	14×14→7×7	256 → 256	
Conv2d(stride=2), GroupNorm, SiLU, Conv2d(stride=1), GroupNorm, SiLU

Latent Projection	12,544	256	
Flattening followed by a Fully Connected Linear mapping

  
Group Normalization (GroupNorm) is favored over standard Batch Normalization to ensure stability across varied batch sizes and to prevent inter-batch statistical leakage during training. The Sigmoid Linear Unit (SiLU) provides a smooth, non-monotonic activation function that preserves small negative gradients, preventing dead neurons in deep networks.

The dimensional capacity of the latent space is tightly constrained to 256 parameters. Exhaustive ablation testing dictates this parameterization. A narrower latent space (e.g., 128 dimensions) induces excessive lossy compression, failing to preserve the intricate crest lines of Martian dune fields during reconstruction. Conversely, expanding the bottleneck to 512 dimensions catastrophically degrades the efficacy of the Phase 2 Isolation Forest. Isolation trees construct decision boundaries by executing random splits across the feature coordinates. If the latent space contains hundreds of dimensions carrying negligible variance, the probability of an isolation tree selecting an informative, high-variance coordinate approaches zero, thereby diluting the isolation depth and rendering the anomaly scoring useless.   

While a Variational Autoencoder (VAE) architecture can be toggled via the configuration dictionary, the deterministic CAE remains the primary engine. The defining characteristic of a VAE—the Kullback-Leibler (KL) divergence penalty—actively pulls all latent embeddings toward a centralized unit Gaussian prior. This probabilistic regularization deliberately shrinks the variance and spatial separation between discrete data clusters. Because the downstream novelty engine relies entirely on this spatial separation to detect outliers, applying a KL penalty fundamentally suppresses the exact distance metrics required for successful anomaly isolation.   

3.2 The Decoder and the Checkerboard Artifact Paradox
The decoder functions as the mirror image of the encoder, accepting the 256-dimensional latent vector, projecting it into a 256×7×7 spatial map, and progressively upsampling it to the original 224×224 resolution. The architectural construction of these upsampling layers is paramount to the validity of the anomaly detection pipeline.   

Historically, deep convolutional generators utilized transposed convolutions (often termed deconvolutions or fractionally strided convolutions) to increase spatial resolution. However, as thoroughly documented by Odena et al. (2016), transposed convolutions utilizing strides greater than one and even kernel sizes suffer from uneven overlapping of the convolutional weights. As the kernel translates across the expanding spatial grid, certain pixels receive multiple overlapping accumulations of gradients, while adjacent pixels receive fewer. In two dimensions, this uneven multiplication creates high-frequency, periodic ringing known as "checkerboard artifacts".   

In standard image generation tasks, minor checkerboard noise is often masked by subsequent adversarial networks or dismissed as negligible. However, this pipeline relies on a precise, pixel-wise subtraction between the original image and the reconstruction to generate error heatmaps in Phase 3. If the decoder inherently stamps a periodic checkerboard grid into every reconstruction, this high-frequency noise will contaminate the error residual, completely obscuring true anomalies beneath a veil of architectural artifacts.   

To permanently eradicate this phenomenon, the decoder entirely circumvents transposed convolutions. Instead, it employs a "Resize-Convolution" paradigm. Spatial upsampling is handled via a mathematically naive, nearest-neighbor interpolation that strictly scales the spatial grid by a factor of 2 (nn.Upsample(scale_factor=2, mode="nearest")). This purely geometric expansion avoids overlapping weight accumulations. The interpolated map is subsequently passed through a standard 3×3 convolutional layer with a stride of 1, effectively smoothing the nearest-neighbor blocks without inducing periodic ringing. This architectural choice is non-negotiable for producing the smooth, structurally faithful reconstructions required for accurate residual analysis.   

3.3 Composite Reconstruction Loss Design
Optimizing the autoencoder solely on Mean Squared Error (MSE) presents a significant vulnerability when processing stochastic textures. The Martian surface is dominated by high-frequency, random phenomena such as expansive regolith, scattered boulders, and aeolian ripples. The mathematical objective of MSE is to minimize the pointwise Euclidean distance between pixels. The path of least resistance for a neural network attempting to minimize MSE on stochastic, unpredictable noise is to simply output the local spatial mean, resulting in a highly blurred, featureless reconstruction. If crater rims and dune crests are blurred away in the reconstruction, they will generate massive residual errors during subtraction. This effectively inflates the baseline error of entirely normal terrain, burying the genuine anomalies under false positives.   

To prevent the network from collapsing into a local-mean predictor, the pipeline enforces a mathematically robust composite loss function that prioritizes structural fidelity, perceptual edge retention, and radiometric accuracy simultaneously. The objective function is defined as:   

L 
total
​
 =λ 
mse
​
 L 
MSE
​
 +λ 
ssim
​
 L 
dSSIM
​
 +λ 
grad
​
 L 
Sobel
​
 
Mean Squared Error (λ 
mse
​
 =1.0): Maintains global intensity matching and baseline radiometric fidelity.   

Structural Similarity Index (λ 
ssim
​
 =0.6): The differentiable SSIM penalty (1−SSIM) is computed over an 11×11 Gaussian sliding window. Unlike MSE, SSIM evaluates the preservation of local spatial statistics—specifically, local means, variances, and covariances. By mathematically rewarding the network for reproducing local contrast structures, SSIM forces the decoder to reconstruct high-frequency textures rather than retreating to blurred averages.   

Sobel Gradient Loss (λ 
grad
​
 =0.3): This penalty acts as a direct, geometry-aware edge preserver. Fixed 3×3 horizontal and vertical Sobel filters are convolved over both the input image and the generated reconstruction to extract high-frequency gradients. The L 
1
​
  norm of the difference between these gradients is calculated. This heavily penalizes the network for mismatched or blurred edges, ensuring that sharp geological discontinuities—such as fault scarps and fresh crater rims—remain crisp in the output.   

By building these perceptual and structural constraints directly from the input data, the pipeline honors the strict requirement avoiding pretrained perceptual networks, ensuring the loss function is uniquely calibrated to Martian topologies.   

4. Latent Space Transformation and Novelty Scoring
Following the completion of the training loop over 120 epochs, the entire HiRISE crop archive is passed through the frozen encoder. This yields a deterministic, 256-dimensional latent representation for every image, alongside an aggregated per-image Mean Squared Error derived from the decoder. This latent dataset forms the empirical foundation for Phase 2: the Isolation Forest novelty engine.   

4.1 Principal Component Analysis and Latent Whitening
As previously established, executing an Isolation Forest directly on a raw, 256-dimensional feature space introduces a severe dilution of isolation depth due to the curse of dimensionality. The architecture resolves this through statistical whitening.   

The raw latent vectors are first standardized (scaled to zero mean and unit variance). Subsequently, Principal Component Analysis (PCA) is applied. The PCA algorithm projects the standardized data into an orthogonal coordinate system, where the axes are ordered by the amount of variance they capture. The transformation is truncated to retain precisely 95% of the total explained variance.   

Simultaneously, a whitening transformation is applied to the retained principal components. Whitening mathematically scales the newly defined axes so that each principal component exhibits unit variance. By compressing the signal into a few dozen orthogonal components and normalizing their variances, the pipeline guarantees that when the subsequent Isolation Forest randomly selects a coordinate for a split, that coordinate will inherently carry meaningful, balanced variance. This completely neutralizes the sampling noise that would otherwise destroy the novelty ranking.   

4.2 Isolation Forest Implementation
The whitened, dimensionality-reduced data is fed into an Isolation Forest architecture initialized with 1,000 constituent estimators. To ensure robust ensemble averaging, the maximum sample size for each tree is capped at 256.   

By default, the scikit-learn implementation of the Isolation Forest's score_samples method returns values where a higher scalar corresponds to a more "normal" data point. Because standard anomaly detection literature expects anomalies to occupy the upper extreme of the scoring spectrum, the output is algebraically inverted:   

Novelty Score=−score_samples(X)
The contamination parameter is strictly left at "auto". Hardcoding a contamination percentage forces the algorithm to act as a binary classifier based on arbitrary human assumptions about the prevalence of anomalies on Mars. By leaving the parameter at "auto", the algorithm acts purely as a relative novelty ranker, deferring the critical task of binary separation to rigorous Extreme Value Theory in the next phase.   

5. Statistical Thresholding via Extreme Value Theory
One of the most complex challenges in unsupervised anomaly detection is establishing a mathematically defensible threshold that separates normal, heavy-tailed data variations from true, exogenous anomalies. Relying on basic Gaussian bounds (e.g., standard deviations from the mean) is invalid, as novelty score distributions are inherently skewed and non-Gaussian. Relying on arbitrary percentiles (e.g., flagging the top 1%) guarantees a fixed number of false positives regardless of the data's actual purity.   

To overcome this, the pipeline applies Extreme Value Theory (EVT), specifically the Peaks-Over-Threshold (POT) methodology utilizing the Generalized Pareto Distribution (GPD).   

5.1 The Peaks-Over-Threshold Method
The central theorem of EVT states that for a broad class of underlying continuous distributions, the exceedances of data points above a sufficiently high anchor threshold u will asymptotically follow a Generalized Pareto Distribution (GPD). This powerful theorem allows the pipeline to model the extreme upper tail of the novelty scores without making hazardous assumptions about the shape of the bulk data distribution below the threshold.   

Given an anchor threshold u, and the subset of novelty scores X where X>u, the exceedances (X−u) are fit to a GPD governed by a shape parameter ξ and a scale parameter σ. The probability of exceeding an even higher extreme threshold t is calculated via the GPD survival function:

P(X>t)=ζ(1+ξ 
σ
t−u
​
 ) 
−1/ξ
 
where ζ=P(X>u) represents the empirical base probability of exceeding the anchor u.   

By algebraically inverting this survival function, the pipeline can derive the precise threshold t required to satisfy a specific, predetermined false-positive budget. The architecture explicitly sets the expected number of false positives across the entire 10,000-crop archive to exactly 0.5. This controlled error rate establishes a stated false-positive budget of under one frame in ten thousand, providing rigorous mathematical confidence in the flagged set.   

5.2 Censored Maximum Likelihood Estimation
Fitting a GPD to contaminated data introduces a profound failure mode. The HiRISE dataset is presumed to contain genuine anomalies. Because these anomalies reside in the extreme upper tail of the novelty scores, passing the entire top decile into a standard Maximum Likelihood Estimation (MLE) optimizer will corrupt the fit. The optimizer will view the massive outlier scores not as anomalies, but as evidence of a fundamentally heavier natural tail, driving the shape parameter ξ upward. This results in an extrapolated threshold t that sits above the maximum observed score, flagging absolutely nothing.   

The pipeline solves this by executing a right-censored Maximum Likelihood Estimation. In this paradigm, a secondary censoring threshold c is introduced. For any observation exceeding c, the exact magnitude of the score is hidden from the optimizer; the algorithm is only informed that the value is greater than c. This restricts the extreme anomalies from dragging the fitted shape parameter, effectively blinding the GPD to the contamination while retaining statistical mass.   

The selection of the censoring quantile c is automated through a plateau-scanning algorithm. The pipeline iteratively increases the censoring quantile and refits the GPD at each step. While c resides beneath the contaminated portion of the tail, the resulting shape parameter ξ forms a horizontal plateau. Once c rises high enough that the true anomalies enter the uncensored pool, ξ violently spikes. The algorithm selects the largest censoring quantile that remains on the stable plateau (defined as the last grid point before ξ deviates from the baseline by more than 0.15), ensuring the GPD fit is derived exclusively from the natural extremes of Martian geology.   

5.3 Fallback Mechanics
In instances where the tail contamination is so severe that a stable plateau cannot be found, the GPD extrapolation becomes mathematically untrustworthy. The pipeline includes explicit logic: if the fitted shape parameter exceeds 0.6, or if the resulting threshold t flags zero crops, a fallback mechanism is triggered.   

The primary fallback is the Kernel Density Estimation (KDE) Valley rule. A Gaussian KDE is fitted over the novelty scores, and the threshold is placed at the first local minimum to the right of the distribution's primary mode. This ensures that if the anomalies form a distinctly separated cluster, the pipeline defaults to density-based separation.   

6. Spatial Interpretability and Geological Hypothesis Generation
Once the statistical threshold isolates the subset of anomalous crops, the architecture shifts from global latent analysis to pixel-wise spatial interpretability. This phase bridges the gap between machine learning and planetary geology, providing actionable insights into why a specific frame was flagged.   

6.1 Pixel-wise Residual Heatmaps
The highest-scoring crops that breach the EVT threshold are retrieved and passed back through the CAE. The raw, unsmoothed squared error between the original input array x and the network reconstruction  
x
^
  is calculated to generate the residual matrix E:   

E=(x− 
x
^
 ) 
2
 
This residual matrix acts as a high-fidelity spatial heatmap, localizing the precise geometric structures that the autoencoder failed to represent. Because the decoder utilizes the Resize-Convolution strategy rather than transposed convolutions, this residual map is free of artificial checkerboard noise, ensuring that every spike in error represents a true morphological deviation.   

6.2 Morphological Descriptors
To facilitate automated hypothesis generation, the spatial geometry of the raw error matrix is reduced to five quantitative descriptors. These descriptors are calculated exclusively on the unsmoothed error to prevent Gaussian blurring from manufacturing artificial spatial autocorrelation.   

Descriptor	Mathematical Interpretation and Function
Concentration Ratio	
The percentage of total squared error captured within the top 1% of pixels, divided by 0.01. A score of 1 implies a perfectly uniform error distribution; a score of 10 indicates high localized peaking.

Gini Coefficient	
Computes the statistical inequality of the error distribution across the entire array.

Moran's I	
Evaluates spatial autocorrelation. High values indicate the error is clustered into coherent, contiguous regions; low values indicate randomized, independent noise.

Structure-Tensor Coherence	
Derived from the eigenvalues of the smoothed spatial gradient tensor. High coherence confirms that the error aligns strongly along a specific vector or axis.

Border Fraction	
The ratio of the total error residing exclusively in the outer 10% perimeter of the image frame.

  
To provide contextual relevance, the raw scalar values of these descriptors are converted into percentiles by comparing them against a baseline sample of 150 ordinary crops drawn from below the anomaly threshold.   

6.3 Geological and Sensor Fault Classifications
The percentile rankings of the morphological descriptors drive an automated reporting mechanism that ties the mathematical error geometry to physical causes on the Martian surface:   

Error Concentrated Along a Linear Feature (High Coherence, High Concentration): This pattern indicates a straight discontinuity that lacks precedent in the training manifold. Natural geological candidates include fault scarps, graben walls, or fresh aeolian ridge crests. If metadata reveals an incongruity with the local sun angle (i.e., the feature casts no shadow where one is mathematically required), the anomaly is likely an artificial splice seam resulting from the joining of distinct orbital passes.   

Error Concentrated in a Compact Region (High Concentration): Points to an isolated morphological structure. On Mars, this is highly indicative of recent surface dynamics, such as fresh impact craters exhibiting highly reflective rayed ejecta, or the presence of seasonal Recurring Slope Lineae (RSL).   

Error Fine-Grained and Spread Over the Frame (Low Moran's I): A lack of spatial autocorrelation indicates that the autoencoder failed to reconstruct localized noise. This is the definitive signature of a sensor-level anomaly, such as a dropped transmission line, data compression artifacts, or general bit-error corruption occurring during telemetry downlink.   

Error Diffuse Across the Frame: If the error is smoothly varying but pervasive, it signals a complete domain shift. The fundamental statistics of the entire crop no longer resemble Martian imagery, raising the strong probability of injected, non-Martian content.   

7. Contextual Metadata Fusion and Geographic Analysis
A comprehensive anomaly detection pipeline must contextualize morphological outliers within their geographic and physical environment. Phase 4 analyzes the telemetry associated with the flagged crops via the source_image_metadata.csv registry.   

7.1 Cross-Referencing Source Observations
Every flagged crop is mapped back to its parent orbital observation (source_image_id). The pipeline executes a one-sided binomial test to determine if any single parent observation contributes a statistically disproportionate number of anomalies compared to the global archive rate.   

Because this test is applied concurrently across hundreds of source images, the Benjamini-Hochberg procedure is enforced to strictly control the False Discovery Rate (FDR). A resulting q-value below 0.05 indicates a statistically significant enrichment. This test is vital for geological interpretation: if the flagged anomalies are scattered uniformly across the entire planet, they are likely random occurrences or transient sensor noise. Conversely, if a massive concentration of anomalies is localized to a single parent image, it suggests the MRO passed over a highly active or highly unusual geological field, warranting immediate scientific review.   

7.2 Multi-modal Anomaly Scoring
As a system extension, the pipeline allows for the direct mathematical fusion of telemetry metadata with the latent image space prior to Isolation Forest scoring.   

The metadata dimensions (sun_angle, season, resolution) are preprocessed into numerical representations (e.g., sine/cosine embeddings for angles, one-hot encodings for discrete seasons). However, simply concatenating a 5-dimensional metadata array with a 256-dimensional image array will result in the metadata being ignored during the randomized splitting of the Isolation Forest.   

To rectify this, the pipeline applies dynamic variance scaling. A scalar multiplier α is calculated such that the total variance of the concatenated metadata block exactly equals 10% of the total variance of the PCA-whitened image block. This calibration ensures that the isolation trees sample the geographic and temporal context frequently enough to identify contextual anomalies—such as a morphologically standard crop that claims to have been captured with a sun angle physically impossible for that specific Martian latitude and season.   

8. System Evolution and Ablation Studies
The final architecture represents the culmination of multiple iterative design phases, documented to provide a rationale for the current configuration.

v1 (Baseline): The initial prototype utilized four stride-2 convolutional stages, a highly compressed 128-dimensional bottleneck, standard transposed convolutions in the decoder, and a pure Mean Squared Error loss. This model failed comprehensively. Reconstructions were smooth and blurry, obliterating dune textures. The latent variance was dominated by arbitrary brightness fluctuations, and the decoder produced catastrophic checkerboard grid artifacts.   

v2 (Loss and Decoder Iteration): Transposed convolutions were entirely replaced with the Resize-Convolution methodology, instantly curing the checkerboard artifacts. The objective function was upgraded to the composite loss (MSE + SSIM + Sobel). This forced the network to preserve sharp crater rims and intricate aeolian textures, transitioning the latent space away from raw brightness and toward structural representation.   

v3 (Statistical rigor): The original pipeline utilized arbitrary percentile thresholds (e.g., the top 1%), which proved highly unstable across random seeds. The introduction of PCA whitening normalized the feature space, stabilizing the Isolation Forest splits. The deployment of the Generalized Pareto Distribution via Peaks-Over-Threshold (POT) replaced arbitrary cutoffs with a statistically rigorous, false-positive-controlled mathematical boundary.   

By successfully uniting deep representation learning, artifact-free convolutional decoding, and extreme value statistical thresholding, this architecture establishes a highly robust, fully unsupervised mechanism for planetary anomaly detection. The pipeline not only scales efficiently to petabyte archives but also transitions seamlessly from abstract latent mathematics to concrete, interpretable geological hypotheses.   

