<div align="center">

<img src="docs/imgs/sense_core_logo.svg" height="140px">

**State-of-the-art Real-time Action Recognition**

---


<!-- BADGES -->
<p align="center">
    <a href="https://20bn.com/">
        <img alt="Documentation" src="https://img.shields.io/website/http/20bn.com.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/TwentyBN/sense/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/TwentyBN/sense.svg?color=blue">
    </a>
    <a href="https://github.com/TwentyBN/sense/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/TwentyBN/sense.svg">
    </a>
    <a href="https://github.com/TwentyBN/sense/blob/master/CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg">
    </a>
</p>

</div>

---

<!-- Add some bullet points for what this repo provides-->

**Note:** Please note that this repo is a fork of the [TwentyBn:Sense](https://github.com/TwentyBN/sense). Models showcased here are finetuned using the pre-trained weights and training tools available from the original repo.

For installation and instructions on runnning the already included demos, please check out the [README](https://github.com/TwentyBN/sense#getting-started) from the main repository. You'll have to also download the pre-trained weights as described there too.


# A Gesture Recognition System for Human Robot Interaction

This repo demonstrates a RGB-based gesture recognition system aimed for Human Robot Interaction. This repository was further integrated [here](https://github.com/alberttjc/hiROS) into the ROS framework with the intention to be used for HRI research.


The following gestures supported are:

## Gestures
<div align="center">

|||||
| ------------- |-------------| ---------| ------------- |
| Come forward                | Handwave           | Pointing       | Rotate Arm Clockwise |
| Come forward                | Pause              |Resume          | Rotate Arm anti-Clockwise |
| Move to the left            | Start              |Thumbs down     | Watch out|
| Move to the right           | Stop               |Thumbs up       |
</div>

<br/>

## Demonstration

<p align="center">
    <img src="docs/gifs/hri_recognition.gif" width="600px">
</p>


Try it yourself: 

```
PYTHONPATH=./ python examples/run_hri_recognition.py --use_gpu
```

---

## Installation Steps & Troubleshooting
Please refer to the original repo for requirements & installation steps [here](https://github.com/TwentyBN/sense#getting-started).

--- 

## License 

The code is MIT but pretrained weights come with a separate license. Please check the original 
[sense repo](https://github.com/TwentyBN/sense) for more information.