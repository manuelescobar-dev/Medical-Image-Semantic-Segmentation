<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
<h3 align="center">Medical Image Semantic Segmentation</h3>

  <p align="center">
    Glomeruli Semantic Segmentation with SegNet-VGG16 on Whole Slide Images (WSI)
    <br />
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

Initially proposed for glomeruli detection in whole slide images, this project aims to create a general framework for the detection of different pathologies in medical images, highlighting the critical need for accurate identification. Glomeruli detection is essential for the prognosis of diabetic kidney disease and various clinical decision-making processes.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

- PyTorch  
- OpenSlide

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

This is an example of how to set up your project locally. To get a local copy up and running, follow these simple steps.

### Installation

1. Clone the repository.
2. We work with Poetry for dependency management. To install the required packages, run:
   ```bash
   poetry install
   ```
3. (Windows) Install OpenSlide. Follow the [instructions](https://openslide.org/).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

1. Indicate the path to the whole slide images in `settings.py`. Raw data (.svs and .xml files) should be stored in the `data/raw` folder.
2. Run the `preprocessing/patching.py` script to divide the whole slide images into patches.
3. Run `semantic_segmentation/train_model.py` to train the semantic segmentation model.
4. To evaluate the model, run `semantic_segmentation/test_model.py`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/manuelescobar-dev/Medical-Image-Semantic-Segmentation.svg?style=for-the-badge
[contributors-url]: https://github.com/manuelescobar-dev/Medical-Image-Semantic-Segmentation/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/manuelescobar-dev/Medical-Image-Semantic-Segmentation.svg?style=for-the-badge
[forks-url]: https://github.com/manuelescobar-dev/Medical-Image-Semantic-Segmentation/network/members
[stars-shield]: https://img.shields.io/github/stars/manuelescobar-dev/Medical-Image-Semantic-Segmentation.svg?style=for-the-badge
[stars-url]: https://github.com/manuelescobar-dev/Medical-Image-Semantic-Segmentation/stargazers
[issues-shield]: https://img.shields.io/github/issues/manuelescobar-dev/Medical-Image-Semantic-Segmentation.svg?style=for-the-badge
[issues-url]: https://github.com/manuelescobar-dev/Medical-Image-Semantic-Segmentation/issues
[license-shield]: https://img.shields.io/github/license/manuelescobar-dev/Medical-Image-Semantic-Segmentation.svg?style=for-the-badge
[license-url]: https://github.com/manuelescobar-dev/Medical-Image-Semantic-Segmentation/blob/master/LICENSE