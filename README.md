**💊 PharmaLens AI**

**Explainable Machine Learning-Based Drug Response Prediction and Pharmaceutical Data Analytics
**


PharmaLens AI is a machine learning-based pharmaceutical analytics platform that predicts experimental drug response using molecular descriptors, drug concentration, and cancer cell-line characteristics.



The project combines XGBoost, RDKit, SHAP, and Streamlit to provide predictive analytics and explainable AI visualizations.



**🌐 Live Application**



[Launch PharmaLens AI](https://pharmalens-ai-tenrnvqcwbapp2zstnesbsi.streamlit.app/)



**💻 GitHub Repository**



[PharmaLens AI Source Code](https://github.com/zeeshanahmedshaik8-a11y/PharmaLens-AI)



**🎯 Project Objectives**

-> Analyze large-scale pharmaceutical drug-response datasets.

-> Extract molecular features from SMILES representations.

-> Develop machine learning models for drug-response prediction.

-> Compare baseline regression with XGBoost.

-> Explain model predictions using SHAP.

-> Deploy an interactive pharmaceutical analytics dashboard.



**🛠️ Technologies Used**



Programming: Python

Data Analysis: Pandas, NumPy

Machine Learning: Scikit-learn, XGBoost

Cheminformatics: RDKit

Explainable AI: SHAP

Visualization: Matplotlib

Web Application: Streamlit

Version Control: Git and GitHub



 **📊 Dataset**



The project uses pharmaceutical screening data from the Broad Institute's PRISM drug-response dataset.



After preprocessing and feature engineering, the machine learning dataset contains approximately 2.58 million observations, covering 4,663 compounds and 568 cell lines.



**🤖 Model Performance**



The models were evaluated using a compound-grouped train/test split to assess generalization to unseen compounds.



<img width="957" height="141" alt="image" src="https://github.com/user-attachments/assets/dc69d539-5eae-4d9b-82b7-63185cb4671f" />




XGBoost achieved the better performance, although its relatively low R² indicates substantial unexplained variation in drug response.



**🔍 Explainable AI**



SHAP analysis helps explain which input features influence model predictions.



Important features include:

-> Hydrogen bond acceptors

-> Molecular weight

-> Topological polar surface area

-> LogP

-> Rotatable bonds

-> Hydrogen bond donors

-> Drug concentration



**🚀 Application Features**


Overview: Pharmaceutical data analytics and visualizations.



Prediction: Machine learning-based drug-response estimation.



Explainability: SHAP feature importance and summary visualizations.



**⚙️ Installation**



Clone the repository:



```bash

git clone https://github.com/zeeshanahmedshaik8-a11y/PharmaLens-AI.git

cd PharmaLens-AI

```



Install dependencies:



```bash

pip install -r requirements.txt

```



Launch the application:



```bash

streamlit run app.py

```



**⚠️ Disclaimer**



PharmaLens AI is an educational and research demonstration. Its predictions have not been clinically validated and must not be used to guide patient treatment or medical decisions.



**🔮 Future Enhancements**



-> Integrate additional biological and genomic features.

-> Improve predictive accuracy using advanced machine learning architectures.

-> Support additional pharmaceutical datasets.

-> Extend explainability to individual predictions.

-> Improve generalization to unseen drugs and cell lines.



**👨‍💻 Author**

Shaik Zeeshan Ahmed

B.Tech – Computer Science and Engineering (Data Science)

GitHub: \[zeeshanahmedshaik8-a11y](https://github.com/zeeshanahmedshaik8-a11y)

