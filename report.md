# Homework 2: Vision Transformer (Variant A)

## 1. Architectural Configurations & Comparison
For this assignment, a custom Vision Transformer (ViT) was implemented from scratch and trained on the CIFAR-10 dataset (32x32 images). Two main configurations were compared, specifically focusing on the impact of the `patch_size`. 

**Configuration 1 (Patch Size = 4):**
* **Patch Size:** 4x4
* **Sequence Length ($N$):** 65 (64 patches + 1 `[CLS]` token)
* **Embedding Dimension ($D$):** 128
* **Trainable Parameters:** ~809,354
* **Self-Attention Cost:** ~540,800 FLOPs per layer
* **Accuracy (after 2 epochs):** 48.87%

**Configuration 2 (Patch Size = 8):**
* **Patch Size:** 8x8
* **Sequence Length ($N$):** 17 (16 patches + 1 `[CLS]` token)
* **Embedding Dimension ($D$):** 128
* **Trainable Parameters:** ~821,642
* **Self-Attention Cost:** ~36,992 FLOPs per layer
* **Accuracy (after 2 epochs):** 51.04%

**Observation on Attention Cost:** The self-attention mechanism scales quadratically with the sequence length $O(N^2 \cdot D)$. By doubling the patch size from 4 to 8, the sequence length dropped drastically (from 65 to 17), which reduced the attention computational cost by over 93%, making the forward pass significantly faster and more memory-efficient.

---

## 2. Error Analysis (Based on Confusion Matrix)
Based on the generated Confusion Matrix for the Patch=8 model, several distinct patterns of misclassification emerged:

1. **Visual and Structural Similarity:** The model heavily confuses **Dogs and Cats**. 253 actual dogs were predicted as cats, and 129 cats were predicted as dogs. This indicates that at early training stages with large patches, the model struggles to differentiate fine-grained features (like snout shape) and instead relies on general structural outlines (four-legged animals).
   Similarly, **Cars and Trucks** are confused (228 actual cars predicted as trucks), likely due to similar rectangular bounding box profiles and shared context (roads).

2. **Background Bias:** An interesting observation is the confusion between **Planes and Ships** (209 planes predicted as ships). Since a patch size of 8 covers a large area (25% of the image width), the model is likely picking up heavily on the dominant background color (blue for both sky and water) rather than the object's defining edges. 

3. **Loss of Fine Details:**
   **Birds** had one of the lowest true positive rates (223) and were scattered across predictions like deer (193), frogs (139), and horses (141). Small objects like birds get compressed into a single 8x8 patch, destroying the spatial resolution needed to identify them properly compared to a smaller patch size like 4x4.

---

## 3. Conclusion on Tokenized Schemes
A flat tokenized scheme (pure ViT) is highly convenient because of its uniform processing and massive scalability. However, its limitations become noticeable when processing small images (like CIFAR-10) with large patch sizes. The lack of inductive biases (like translation invariance found in CNNs) means the ViT requires massive amounts of data and longer training to inherently learn edge detection and local textures. When limited to small datasets or fewer epochs, it tends to over-rely on global features like background color or large geometric shapes.
