<a name="readme-top"></a>

[![LinkedIn][linkedin-shield]][linkedin-url]
[![GitHub][github-shield]][github-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/wicklyg/fcm-pso">
    <img src="img/img.webp" alt="Logo" width="120" height="120">
  </a>
  
  <h3 align="center">Clustering of Regencies/Cities in Indonesia Based on Welfare Indicators Using Fuzzy C-Means and Particle Swarm Optimization</h3>

  <p align="center">
    This project aims to cluster regencies/cities in Indonesia based on welfare indicators using the Fuzzy C-Means (FCM) algorithm and Particle Swarm Optimization (PSO).
    <br />
    <a href="https://github.com/wicklyg/fcm-pso"><strong>Explore the docs »</strong></a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    <li>
      <a href="#installation">Installation</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#setup">Setup</a></li>
      </ul>
    </li>
    <li><a href="#running-the-project">Running the Project</a></li>
    <li><a href="#data-sources">Data Sources</a></li>
    <li><a href="#futures-work">Futures Work</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

This project aims to cluster regencies/cities in Indonesia based on welfare indicators using the Fuzzy C-Means (FCM) algorithm and Particle Swarm Optimization (PSO). The goal is to provide insightful clustering results that can guide development plans and policies to improve welfare in different regions.

Here's why this project is valuable:

- **Insightful Clustering**: The project helps in identifying similar regencies/cities based on multiple welfare indicators.
- **Policy Guidance**: The clustering results can guide policymakers in focusing their efforts on regions with similar needs.
- **Enhanced Algorithms**: By combining FCM with PSO, the project aims to improve the clustering results.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Installation

### Prerequisites

Ensure you have Python installed on your system. The project requires the following Python packages:

- numpy
- pandas
- matplotlib
- scikit-learn
- openpyxl

### Setup

1. Clone the repository to your local machine:
   ```bash
   git clone <repository_url>
   cd project_folder
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>
         ```

## Running the Project

1. Ensure your dataset `Data IKR New.xlsx` is placed inside the `data` directory.
2. Run the main script:
   ```bash
   python main.py
   ```
3. The processed data will be saved to `Processed_Data.xlsx` in the data directory.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Data Sources

The primary dataset used in this project is [Data IKR New.xlsx](https://github.com/gubuktani/ml-models/edit/main/README.md), which contains various welfare indicators for regencies/cities in Indonesia. The indicators include health, education, and economic metrics.

1. Health Indicators:
   - Morbidity Rate (Angka Kesakitan)
   - Life Expectancy (Angka Harapan Hidup)
   - Households with Access to Clean Water (Persentase Rumah Tangga dengan Akses Air Bersih)
   - Households with Proper Sanitation (Persentase Rumah Tangga dengan Akses Sanitasi Layak)
2. Education Indicators:

   - Literacy Rate (Angka Melek Huruf)
   - School Participation Rate (Angka Partisipasi Sekolah)
   - Expected Years of Schooling (Harapan Lama Sekolah)

3. Economic Indicators:
   - Gini Index (Indeks Gini)
   - Percentage of Poor Population (Persentase Penduduk Miskin)
   - Open Unemployment Rate (Tingkat Pengangguran Terbuka)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Future work

Potential future work includes:

- Enhancing the PSO algorithm with adaptive parameters.
- Exploring additional clustering algorithms.
- Integrating more welfare indicators.
- Applying the model to different datasets for broader analysis.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

- Data provided by [Badan Pusat Statistik](https://www.bps.go.id/id).

## References

1. Bezdek, J. C., R. Ehrlich, dan W. Full. (1984). FCM: The Fuzzy C-Means Clustering Algorithm. _Computer & Geosciences_.
2. Izakian, H. dan A. Abraham. (2011). Fuzzy C-Means and Fuzzy Swarm for Fuzzy Clustering Problem. _Expert Systems with Applications_.
3. Badan Pusat Statistik (BPS) - Various data sources for welfare indicators.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/wicklygusthvi
[github-shield]: https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white
[github-url]: https://github.com/wicklyg
