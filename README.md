# Comprehensive Notes on Convolutional Neural Networks (CNNs) & Deep Learning

## 1. Why Replace Large Filters (e.g., $5\times 5$, $7\times 7$) with Stacked $3\times 3$ Filters?

### 1.1 Spatial Receptive Field Equivalence

* **Two stacked** $3\times 3$ **conv layers** (with stride $S=1$) have an effective receptive field equal to a single $5\times 5$ conv layer:
  

  $$
  \text{Receptive Field} = 3 + 3 - 1 = 5
  $$

* **Three stacked** $3\times 3$ **conv layers** (with stride $S=1$) have an effective receptive field equal to a single $7\times 7$ conv layer:
  

  $$
  \text{Receptive Field} = 3 + 3 + 3 - 2 = 7
  $$

### 1.2 Advantages of Using Multiple Small Filters

1. **Fewer Parameters (Parameter Reduction):**

   * Assume input and output channels are both $C$:

     * **Single** $7\times 7$ **layer:**
       

       $$
       \text{Params} = 7 \times 7 \times C \times C = 49 C^2
       $$

     * **Three stacked** $3\times 3$ **layers:**
       

       $$
       \text{Params} = 3 \times (3 \times 3 \times C \times C) = 27 C^2
       $$

     * **Savings:**
       

       $$
       \frac{49 C^2 - 27 C^2}{49 C^2} \approx 44.9\% \text{ parameter reduction}
       $$

     * **Single** $5\times 5$ **layer:**
       

       $$
       5 \times 5 \times C \times C = 25 C^2
       $$

       
       vs **Two stacked** $3\times 3$ **layers:**
       

       $$
       2 \times (3 \times 3 \times C \times C) = 18 C^2 \quad (28\% \text{ reduction})
       $$

2. **More Non-Linearity:**

   * Stacked layers feature activation functions (e.g., ReLU) after *each* $3\times 3$ convolution:
     

     $$
     \text{Conv} \rightarrow \text{ReLU} \rightarrow \text{Conv} \rightarrow \text{ReLU} \rightarrow \text{Conv} \rightarrow \text{ReLU}
     $$

   * Stacking multiple non-linear functions allows the network to represent significantly more complex composite functions than a single large non-linear layer.

3. **Disadvantage:**

   * **Higher Memory Consumption:** Storing intermediate activation feature maps during the forward pass and backpropagation requires more VRAM/RAM.

## 2. Spatial Dimension Rules & Padding Strategy

### 2.1 Rules of Thumb for Feature Map Sizes

1. **Preserve Spatial Resolution in Conv Layers:** Use padding ($P$) to keep spatial dimensions constant during convolution so feature extraction occurs without losing spatial edge pixels.

2. **Downsample Exclusively in Pooling Layers:** Let Pooling layers handle spatial dimension reduction.

3. **Power of 2 Image Dimensions:** Input dimensions (width and height) should preferably be powers of $2$ (e.g., $32, 64, 128, 224, 256$) for seamless division during downsampling operations (e.g., $224 \xrightarrow{Pool} 112 \xrightarrow{Pool} 56 \xrightarrow{Pool} 28 \xrightarrow{Pool} 14 \xrightarrow{Pool} 7$).

### 2.2 Padding Calculations (Same Padding Step-by-Step Derivation)

To ensure Output Size ($O$) equals Input Size ($I$) when Stride $S = 1$:

$$
O = \frac{I - F + 2P}{S} + 1
$$

Setting $O = I$ and $S = 1$:

$$
I = \frac{I - F + 2P}{1} + 1
$$

$$
I = I - F + 2P + 1
$$

$$
I - I + F - 1 = 2P \implies 2P = F - 1 \implies P = \frac{F - 1}{2}
$$

* For $F = 3\times 3 \implies P = \frac{3-1}{2} = 1$

* For $F = 5\times 5 \implies P = \frac{5-1}{2} = 2$

* For $F = 7\times 7 \implies P = \frac{7-1}{2} = 3$

### 2.3 Example Calculation

Given Input: $32 \times 32 \times 3$, Conv layer with $F = 3\times 3$, $P = 1$, $S = 1$, and $N = 16$ filters:

$$
\text{Output Width/Height} = \frac{32 - 3 + 2(1)}{1} + 1 = 32
$$

* **Output Tensor Volume:** $32 \times 32 \times 16$

## 3. Early Downsampling & Memory Trade-offs

* **The Memory Problem:** Keeping large spatial dimensions (e.g., $224 \times 224$) with many channels in early layers consumes massive activation memory (e.g., $\approx 72\text{ MB}$ per image batch).

* **Edge Information Protection:** Padding prevents border pixels from being washed out prematurely because filters overlap border region inputs across multiple strides rather than skipping them.

* **The Solution / Trade-off:** Sacrificing some spatial resolution in the very first layers using early downsampling (e.g., $7\times 7$ conv with Stride 2 or Max Pooling) reduces memory usage, activation storage, and gradient computation overhead.

## 4. Modern CNN Architectural Innovations

```
Sequential: [Input] -> [Conv] -> [ReLU] -> [Pool] -> [FC] -> [Output]

Inception:           /-- [Conv 1x1] --\
          [Input] --+-- [Conv 3x3] ----+-- [Concat] -> [Output]
                     +-- [Conv 5x5] ----+
                     \-- [Max Pooling] -/

ResNet:   [Input x] ----+---------------------(+) -> [Output y = F(x) + x]
                        |                      ^
                        \-- [Conv] -> [ReLU] --/

```

### 4.1 Traditional Sequential Architecture

* Standard linear pipeline: `Input -> [Conv -> ReLU -> Pool] -> FC -> Output` (e.g., AlexNet, VGG).

### 4.2 Inception Architecture (GoogleNet)

* **Multi-branch Architecture:** Processes features across multiple filter scales ($1\times 1, 3\times 3, 5\times 5$, and Pooling) in parallel within the same block, concatenating their outputs depth-wise.

* $1\times 1$ **Convolutions:** Used for dimensionality reduction before expensive $3\times 3$ and $5\times 5$ convolutions to lower compute overhead.

### 4.3 Residual Networks (ResNet)

* Introduces **Skip Connections (Shortcuts)** to solve the vanishing/exploding gradient problem in very deep networks.

* Formulate learning target as residual mapping:
  

  $$
  y = F(x) + x
  $$

* Allows gradients to flow directly through skip connections during backpropagation without degradation.

## 5. Fundamentals of Neural Networks (ANN Basics)

```
[Layer 1: 784 Neurons]       [Hidden Layer]       [Output Layer: 10 Neurons]
  (28x28 Pixels)               (Sub-patterns)             (Class Scores)
     ( O ) ----------------------> ( O ) ---------------------> [ 0 ]
     ( O ) ----------------------> ( O ) ---------------------> [ 1 ]
     ( O ) ----------------------> ( O ) ---------------------> [ ... ]
     ( O ) ----------------------> ( O ) ---------------------> [ 9 ]

```

### 5.1 Neurons and Activations

* **Neuron:** Holds a numerical scalar value called an **Activation**.

* **MNIST Example:** Image of $28 \times 28 = 784$ pixels.

  * **Input Layer:** $784$ neurons (representing pixel grayscale values from $0.0$ to $1.0$).

  * **Output Layer:** $10$ neurons (class scores for digits $0-9$).

  * **Hidden Layers:** Extract hierarchy of features (edges $\rightarrow$ loops/sub-patterns $\rightarrow$ complete digits).

### 5.2 Mathematical Formulation

For a layer $l$ receiving input vector $a^{(l-1)}$:

$$
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
$$

$$
a^{(l)} = \sigma(z^{(l)})
$$

Where:

* $W^{(l)}$: Weight matrix of connections.

* $b^{(l)}$: Bias vector.

* $\sigma(\cdot)$: Non-linear activation function.

### 5.3 Common Activation Functions

1. **ReLU (Rectified Linear Unit):**
   

   $$
   \text{ReLU}(a) = \max(0, a)
   $$

   * Fast to compute, prevents vanishing gradients for positive inputs.

2. **Sigmoid:**
   

   $$
   \sigma(x) = \frac{1}{1 + e^{-x}}
   $$

   * Squeezes real numbers into $(0, 1)$. Mainly used in legacy models or binary output layers.

## 6. From Fully Connected (FC) Networks to CNNs

### 6.1 Scaling Limitations of FC Layers

* Fully Connected networks fail to scale to high-resolution images.

* **CIFAR-10 Example (**$32 \times 32 \times 3$**):** Single neuron in hidden layer requires $3,072$ weights.

* **Large Image Example (**$200 \times 200 \times 3$**):** Single neuron requires $200 \times 200 \times 3 = 120,000$ weights!

* **Result:** Parameter explosion leads to severe overfitting and massive computational burden.

### 6.2 Key Properties of Convolutional Layers

1. **3D Volume of Activations:** Layers have Width, Height, and Depth (channels).

2. **Local Connectivity (Receptive Field):** Neurons connect only to a small local spatial region of the input volume.

3. **Parameter Sharing:** Neurons in the same depth slice share the same weight filter matrix and bias, assuming feature usefulness is spatially invariant.

## 7. Convolutional Layer Mechanics & Hyperparameters

### 7.1 Key Spatial Hyperparameters

1. **Filter Size / Receptive Field (**$F$**):** Spatial dimensions of filter (e.g., $3\times 3, 5\times 5$).

2. **Depth (**$K$ **/** $N$**):** Number of filters applied; defines the output depth.

3. **Stride (**$S$**):** Spatial step size when sliding the filter across the input.

4. **Zero-Padding (**$P$**):** Number of zero layers added around image borders.

### 7.2 Output Spatial Size Formula

Given Input $W_1 \times H_1 \times D_1$:

$$
W_2 = \frac{W_1 - F + 2P}{S} + 1
$$

$$
H_2 = \frac{H_1 - F + 2P}{S} + 1
$$

$$
D_2 = K
$$

### 7.3 Detailed AlexNet Layer 1 Parameter Calculations

* **Input Volume:** $227 \times 227 \times 3$

* **Hyperparameters:** Filter size $F = 11 \times 11$, Stride $S = 4$, Padding $P = 0$, Number of Filters $K = 96$

* **Output Volume Calculation:**
  

  $$
  W_2 = \frac{227 - 11 + 2(0)}{4} + 1 = \frac{216}{4} + 1 = 54 + 1 = 55
  $$

  * **Output Shape:** $55 \times 55 \times 96$

  * **Total Output Neurons:** $55 \times 55 \times 96 = 290,400 \text{ neurons}$

* **Parameter Count Comparison:**

  * **Without Parameter Sharing:**
    Each neuron has its own independent weights ($11 \times 11 \times 3 = 363$ weights + $1$ bias = $364$ parameters):
    

    $$
    \text{Total Params} = 290,400 \times 364 = 105,705,600 \text{ parameters } (\approx 105.7 \text{ Million})
    $$

  * **With Parameter Sharing:**
    All $55 \times 55$ neurons in each depth map share the exact same $11 \times 11 \times 3$ filter weights and single bias:
    

    $$
    \text{Total Params} = 96 \times (11 \times 11 \times 3 + 1) = 96 \times 364 = 34,944 \text{ parameters}
    $$

## 8. Backpropagation & Parameter Sharing Gradient Rules

* When parameter sharing is used, multiple spatial locations share the exact same filter weights $W$.

* **Gradient Accumulation Rule:** During backpropagation, the gradient of the loss function $L$ with respect to a shared weight $w$ is the **sum** of gradients across all spatial positions $(x, y)$ that used that weight:
  

  $$
  \frac{\partial L}{\partial w} = \sum_{x, y} \left. \frac{\partial L}{\partial w} \right\vert{}_{(x, y)}
  $$

* An optimizer then uses this accumulated gradient to update the shared weight parameter.

## 9. Matrix Multiplication Implementation (`im2col`)

To perform high-speed convolutions using optimized BLAS linear algebra libraries, convolutions are cast as matrix multiplications:

1. **Extract Blocks (`im2col`):** Each local patch ($F \times F \times D_{in}$) sliding across the input volume is flattened into a column vector of size $(F \cdot F \cdot D_{in}) \times 1$.

2. **Construct Input Matrix:** Repeating this across all spatial locations ($W_{out} \cdot H_{out}$) yields a matrix of shape:
   

   $$
   \left( F \cdot F \cdot D_{in} \right) \times \left( W_{out} \cdot H_{out} \right)
   $$

3. **Construct Filter Matrix:** Flatten $K$ filters into a matrix of shape:
   

   $$
   K \times \left( F \cdot F \cdot D_{in} \right)
   $$

4. **Matrix Product:** Multiply Filter Matrix by Input Matrix to yield output shape:
   

   $$
   K \times \left( W_{out} \cdot H_{out} \right)
   $$

5. **Reshape:** Reshape output into 3D volume $W_{out} \times H_{out} \times K$.

### 9.1 `im2col` Worked Example (AlexNet Layer 1)

* **Input Volume:** $227 \times 227 \times 3$

* **Filter Size:** $11 \times 11 \times 3 = 363$

* **Number of Spatial Locations:** $55 \times 55 = 3,025$

* **Constructed Input Matrix:** $363 \times 3,025$

* **Constructed Filter Matrix:** $96 \times 363$

* **Matrix Multiplication Output:** $(96 \times 363) \times (363 \times 3,025) = 96 \times 3,025$

* **Reshaped Volume:** $55 \times 55 \times 96$

## 10. Pooling & Normalization Layers

### 10.1 Pooling Layers

* **Purpose:** Reduce spatial size, cut compute cost/parameters, introduce scale/translation invariance, and control overfitting.

* **Max Pooling:** Extracts maximum activation value in each local window.

  * *Standard Configuration:* $F = 2, S = 2$ (downsamples spatial area by $75\%$).

#### Matrix Example of Max Pooling ($2\times 2$, Stride 2)

$$
\begin{bmatrix} 1 & 1 & 2 & 4 \\ 5 & 6 & 7 & 8 \\ 3 & 2 & 1 & 0 \\ 1 & 2 & 3 & 4 \end{bmatrix} \xrightarrow{\text{Max Pool } 2\times 2, S=2} \begin{bmatrix} 6 & 8 \\ 3 & 4 \end{bmatrix}
$$

* **Average Pooling:** Computes average value in window.

* **Output Formulas:**
  

  $$
  W_2 = \frac{W_1 - F}{S} + 1, \quad H_2 = \frac{H_1 - F}{S} + 1, \quad D_2 = D_1
  $$

### 10.2 Normalization Layers

* **Local Response Normalization (LRN):** Historically used in AlexNet to mimic biological lateral inhibition between adjacent channels.

* **Modern Preference:** LRN has been superseded by **Batch Normalization (BatchNorm)** and **Layer Normalization**, as LRN did not contribute significantly to efficiency or accuracy gains compared to modern normalization layers.

## 11. Converting Fully Connected Layers to Convolutional Layers (FC $\rightarrow$ Conv)

### 11.1 Functional Equivalence

A Fully Connected layer and a Conv layer both compute dot products. Thus, any FC layer can be converted into a Convolutional layer.

### 11.2 Conversion Procedure

To convert an FC layer with $N_{out}$ neurons receiving input feature map $W \times H \times C$:

* Replace FC layer with a Conv layer having:

  * Filter Size $F = W \times H$

  * Padding $P = 0$, Stride $S = 1$

  * Number of Filters $K = N_{out}$

### 11.3 Conversion Example & Parameter Comparison (AlexNet FC1)

* **Input to FC1:** $7 \times 7 \times 512$ (Flattened dimension = $7 \times 7 \times 512 = 25,088$)

* **FC Output:** $4,096$ neurons

* **Standard FC Layer Parameters:**
  

  $$
  \text{Weights} = 25,088 \times 4,096 = 102,760,448
  $$

  $$
  \text{Biases} = 4,096
  $$

  $$
  \text{Total FC Params} = 102,760,448 + 4,096 = 102,764,544 \text{ parameters } (\approx 102.7 \text{ Million})
  $$

* **Equivalent Conv Layer:**

  * Filters: $K = 4096$, Size $F = 7 \times 7$, Padding $P = 0$, Stride $S = 1$.

  * **Output Volume:** $1 \times 1 \times 4,096$.

### 11.4 Benefit: Efficient Sliding Window / Full Image Evaluation

Instead of manually cropping an oversized image (e.g., $384 \times 384$) into multi-crop inputs ($224 \times 224$):

#### Multi-crop Approach Calculations:

* Number of crops along one axis with step stride $32$:
  

  $$
  \text{Crops per axis} = \frac{384 - 224}{32} + 1 = \frac{160}{32} + 1 = 5 + 1 = 6
  $$

* Total crops required:
  

  $$
  \text{Total Crops} = 6 \times 6 = 36 \text{ separate forward passes}
  $$

#### Fully Convolutional Single-Pass Approach Calculations:

* Input large image $384 \times 384 \times 3$ directly into the network.

* Spatial downsampling factor of initial Conv/Pool layers is $32$ ($2^5 = 32$).

* Feature map size entering converted FC layer:
  

  $$
  \text{Feature Map Size} = \frac{384}{32} \times \frac{384}{32} \times 512 = 12 \times 12 \times 512
  $$

* Applying converted $7 \times 7$ Conv filter ($S=1, P=0$):
  

  $$
  H_{out} = \frac{12 - 7}{1} + 1 = 6
  $$

* **Final Class Score Volume:** $6 \times 6 \times 1000$ (representing evaluation over all 36 crop spatial positions in a single forward pass without duplicate compute).

## 12. Standard CNN Architectural Patterns

### 12.1 Canonical Sequence

```
INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC

```

* Usually $N \le 3$, $M \ge 1$, $K < 3$.

* Filter sizes are typically small ($K = 3\times 3$ or $5\times 5$).

### 12.2 Design Philosophy

* Prefer multiple consecutive small Conv layers with non-linearities before applying downsampling Pooling layers.

* Stacking convolutions preserves spatial features while building deeper, richer abstraction levels.

## 13. Worked Example: Tracking Feature-Map Dimensions Through a CNN

### 13.1 Convolution Preserving Spatial Dimensions

* **Input:** $32 \times 32 \times 3$

* **Convolution:** $3 \times 3$, $K=16$ filters, stride $S=1$, padding $P=1$

* **Output Dimensions:**
  

  $$
  W_2 = \frac{32-3+2(1)}{1}+1=32
  $$

  $$
  H_2 = \frac{32-3+2(1)}{1}+1=32
  $$

* **Output Shape:** $32 \times 32 \times 16$

### 13.2 Max Pooling Reducing Spatial Dimensions

Apply Max Pooling with $F=2$ and $S=2$, assuming no padding ($P=0$):

$$
W_2 = \frac{32-2}{2}+1=16
$$

$$
H_2 = \frac{32-2}{2}+1=16
$$

* **Input Shape:** $32 \times 32 \times 16$

* **Output Shape:** $16 \times 16 \times 16$

## 14. Convolution Parameter Count Including Bias

$$
\text{Parameters} = K \times (F \times F \times D_{in} + 1)
$$

### 14.1 Worked Example

* Filter size: $3 \times 3$

* Input channels: $D_{in}=3$

* Number of filters: $K=16$
  

  $$
  \text{Parameters} = 16 \times (3 \times 3 \times 3 + 1) = 16 \times 28 = 448
  $$

## 15. Feature Extraction Versus Classification

### 15.1 Feature Extraction

* Early layers: Edges, simple color contrasts, and textures.

* Intermediate layers: Combinations of edges and textures (corners, loops).

* Deeper layers: Complex object parts and task-relevant representations.

### 15.2 Classification Head

* Converts extracted features to class scores.

* Softmax layer converts logits into normalized probabilities:
  

  $$
  p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
  $$

## 16. How a CNN Learns: The Training Loop

1. **Forward Pass:** Feed image to get predicted scores.

2. **Loss Calculation:** Compute cross-entropy loss between predictions and true labels.

3. **Backpropagation:** Compute gradients of loss relative to parameters.

4. **Optimizer Update:** Update parameters using SGD or Adam:
   

   $$
   \theta_{\text{new}} = \theta_{\text{old}} - \eta \nabla_\theta L
   $$

5. **Repeat:** Process across batches and epochs.

## 17. RGB Images as Numeric Tensors

* RGB image shape in channel-last convention: $H \times W \times 3$.

* Pixel values ($0 - 255$) normalized to $[0, 1]$:
  

  $$
  x_{\text{scaled}} = \frac{x}{255}
  $$

## 18. Fully Connected Neuron Mathematical Recipe

$$
z = \sum_{i=1}^{n} w_i x_i + b \implies z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}
$$

$$
a = \sigma(z) \implies a^{(l)} = \sigma(z^{(l)})
$$

## 19. Summary Visual Concepts

* **Stacked** $3\times 3$ **Filters:** Build larger receptive fields with fewer parameters and added non-linearities.

* **Inception Branches:** Parallel operations concatenated along channel depth.

* **ResNet Skip Connections:** Direct residual gradient highway $y = F(x) + x$.

* **`im2col` Transformation:** High-speed BLAS matrix multiplication for convolutions.

* **Fully Convolutional Conversion:** Eliminates redundant computations in multi-crop evaluation over large images ($384\times 384 \rightarrow 6\times 6\times 1000$).
