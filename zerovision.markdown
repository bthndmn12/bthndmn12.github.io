---
layout: page
title: Zerovision
permalink: /zerovision/
image: /assets/images/20240201174519
---

# {{ page.title }}

The ZeroVision project aims to develop a deep learning-based system for autonomously segregating foam and particles on the surface of wastewater. This initiative focuses on monitoring the contamination of coarse grates filtering wastewater before it enters the treatment plant and measuring the size and distribution of bubbles in secondary settling tanks to indicate the efficiency of biological treatment.

Segmentation is a primary step in this measurement process. We employ Meta's state-of-the-art model, the Segment Anything Model (SAM), for this phase. Initially, we encountered challenges in fine-tuning the model. Despite the original SAM paper and Meta's GitHub repository suggesting that fine-tuning was not directly supported, we sought alternative approaches.

The model excels in detecting particles, foam, and bubbles on the water's surface. However, based on literature suggesting fine-tuning and Federated Learning could enhance performance, we experimented with these methods. Our experiments with specific techniques mentioned in the Literature Analysis section and testing of SAM model variants yielded generally successful results.

{% include hfspace.html %}



## Project Documentation

Below is a list of documentation available for the ZeroVision project:


{% include sidebar.html %}


