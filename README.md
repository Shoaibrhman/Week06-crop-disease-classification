\# Week 6: Crop Disease Classification from Leaf Photos with ResNet-18



Project 2, Week 6 of \*Introduction to Applied AI\* (BS 8th Semester, Department of Electronics).

Kaggle notebook: https://www.kaggle.com/code/shoaib1830/week-6-crop-disease



\## Problem

\*\*Who benefits:\*\* farmers and agricultural extension workers who need a fast first check of leaf disease.

\*\*Why it matters:\*\* the FAO estimates that up to 40% of global crop production is lost to plant pests and diseases every year, and diseases spotted late cost farmers part of the harvest.



\## Datasheet

\- \*\*Dataset:\*\* PlantVillage (abdallahalidev), https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset. Only the `color` version is used; the `grayscale` and `segmented` folders hold the same images, and mixing them would leak the same leaf into train and test.

\- \*\*Licence:\*\* CC BY-NC-SA 4.0

\- \*\*Labelled by:\*\* not stated on the Kaggle page

\- \*\*Composition:\*\* 54,305 RGB images (256 x 256 pixels) in 38 classes covering 14 crops, healthy and diseased leaves. Classes are strongly imbalanced: from 5,507 images (Orange Huanglongbing) to 152 (Potato healthy), a ratio of \*\*36.2 : 1\*\*.

\- \*\*Collection:\*\* single leaves on plain backgrounds under controlled lighting (lab-style photos, not field photos).

\- \*\*Uses and limits:\*\* a research and teaching dataset. It should not be used to make real spraying or treatment decisions.



\## Data audit

| Check | Result |

|---|---|

| Image size and mode | all sampled images 256 x 256, RGB |

| Exact duplicates (MD5) | 21 groups of identical files, all inside the training folder, no label conflicts |

| Near-duplicates between train and the test set | about 8 of 7,543 test images (1 hit in a sample of 1,000) |

| Group split | every image is its own group (1.00 images per group, the file names carry no plant id); groups shared by train and val: 0, by train and test: 0 |

| Split sizes | train 37,711 / validation 9,051 / test 7,543 |



\*\*First shortcut suspect:\*\* background and lighting. Using only four 8 x 8 corner patches per image, a random forest predicted the class with 0.309 accuracy (chance 0.026).



\## Method

\- ResNet-18 pretrained on ImageNet, 224 x 224 input, ImageNet normalisation.

\- Augmentation on the training set only: random resized crop, flips, small rotation, mild brightness and contrast jitter (colour is evidence of disease, so no strong hue changes).

\- Imbalance: weighted cross-entropy with weights N / (K x n\_c), from 0.260 (Orange Huanglongbing) to 9.105 (Potato healthy).

\- Phase 1: new 38-class head only (19,494 trainable parameters), 2 epochs. Phase 2: head plus `layer4`, 3 epochs. The best epoch is chosen by validation \*\*macro-F1\*\*, not accuracy. Training took about 130 s per epoch on a T4.



\## Results (test set, used once)

| Measure | Value |

|---|---|

| Validation macro-F1, head only | 0.901 |

| Validation macro-F1, after fine-tuning | 0.978 |

| \*\*Test macro-F1\*\* | \*\*0.979\*\* |

| Test accuracy | 0.986 (105 errors in 7,543 images) |

| Weakest recall | Potato healthy 0.920 (25 images), Corn Northern Leaf Blight 0.921, Tomato Early blight 0.922 |

| Most frequent confusion | Corn Northern Leaf Blight predicted as Corn Cercospora leaf spot (11 images) |



This is a 38-class task, so the threshold, sensitivity, specificity and PPV analysis (binary tasks) does not apply.



\## Grad-CAM and shortcut check

\- In the six apple examples inspected (three correct, three wrong) the heat sits on the leaf and its lesions, not on the background.

\- A border-only test (only the outer 12% frame visible) still scored \*\*0.238 accuracy against chance 0.026\*\* (full image: 0.986), so background and lighting carry some class information. Verdict: shortcut suspected, not confirmed.

\- Next test: evaluate on field photos from another source and on the dataset's `segmented` version.



\## Limitations

PlantVillage photos are single leaves on plain backgrounds, so 0.979 is a lab-condition score and overstates field performance (published results show a large drop on photos from other sources). Every image is its own group, so photos of the same leaf could sit on both sides of the split. The rarest classes have few test images, so their scores are noisy.



\## Responsible use

\- \*\*Who could be harmed, and how?\*\* A farmer whose real disease is missed (it spreads) or who is told the wrong disease and sprays the wrong chemical: lost harvest, wasted money, avoidable chemical use.

\- \*\*Where was it NOT validated?\*\* Field photos with natural backgrounds, phone cameras and varied lighting, crops and varieties that are not in the dataset, and local conditions.

\- \*\*What is the human role?\*\* An advisory first check that helps prioritise which plants an extension worker or agronomist inspects; never a final diagnosis or an automatic treatment decision.

\- \*\*What evidence would change my conclusions?\*\* External validation on field photos from local farms with per-class recall, and confirmation that Grad-CAM and the border-only test show the model uses the leaf, not the background.



\## Files

\- `week-6-crop-disease.ipynb`: the full notebook with outputs and written answers

\- `week6\_model.pt`: trained weights (with class names, backbone and image size stored inside)

