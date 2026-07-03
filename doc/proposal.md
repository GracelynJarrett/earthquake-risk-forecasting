# Project Proposal — Earthquake Risk Forecasting

## What

Earthquakes can be detected by seismic waves underground, but predicting their size and timing is very tricky. This system is designed to help emergency managers predict the likelihood of a large earthquake (4.5 or higher) occurring within the next week. California, Japan, and Greece are the main locations for this project. Using XGBoost, a gradient-boosted classifier, the system outputs a weekly risk score for each region and visualizes it via an interactive dashboard for disaster preparedness teams.

## Why

Right now, there are many fires throughout Utah, and that got me thinking of what other natural disasters Utah is prone to. Earthquakes are one of them. Utah is on a fault line, meaning there tend to be more earthquakes than in other areas. After some research, I found that many AI models try to predict when an earthquake will occur; however, many models exhibit data leakage, giving them non-existent data and boosting the likelihood of an earthquake from 23% to 95%. I want to build an ML model that doesn’t suffer from data leakage and can predict the likelihood of a large earthquake (magnitude 4.5 or higher) occurring in the next week.

## Your Takeaway

From this project, I hope to gain more experience working with datasets that could have many leaks or biases by implementing a strict temporal train/validation/test split. I want to understand Airflow’s capabilities and how I can use it to manage my system. I want to prove to myself that I can independently take a project from raw data to a deployed, live system, and that I have the skills to understand and create AI Systems.

## Tech Stack

| Tool / Technology | Description | Familiarity |
|---|---|---|
| USGS | A free data source with earthquake data that updates every minute. I have not worked with this dataset before, but I plan to explore the documentation and run a test pull in week 1 | Never used before |
| MLflow | A system to help monitor Model Accuracy, and can be used to compare different models to one another. | Familiar |
| Airflow | A pipeline orchestration tool that schedules and automates data pipelines, such as pulling fresh data. I will follow official documentation and build a simple pipeline in week 4 | Unfamiliar (only used once) |
| Next.js | A system like Streamlit, but it looks nicer. I have never used this by myself. I want to get experience with it because I plan to use it in my Capstone. To gain experience, I will follow tutorials before week 5 | Unfamiliar (only used in group project) |
| XGBoost | Classifier model for tabular structured data | Familiar |
| Python | Core code language | Familiar |
| pandas | Data manipulation and feature engineering | Familiar |
| Scikit-learn | Preprocessing, train/validate/test splits, evaluation metrics | Familiar |
| GitHub | Helps keep everything organized and grants easy access for others | Familiar |
| Logistic regression | Baseline model that lives inside Scikit-learn | Familiar |
| Jupyter notebooks | Required for the project to analyze the dataset | Familiar |
| SQLite | A lightweight, file-based database that stores the cleaned USGS data so my pipeline reads from one consistent source, kept in the `data/` folder. It is built into Python and works directly with pandas. I will follow the Python `sqlite3`/pandas documentation to get up to speed. | New to me |

## Schedule

**Week 1:** focus on coming up with a project idea, researching the project to see if it’s possible, and writing a proposal for the project. Having a strong foundation for the project will guide me through the development process and clarify the target audience. By the end of the week, I will have a project idea, a completed proposal with a schedule, and my pitch prepared for Thursday. I'll be behind schedule if my proposal isn’t done or approved.

**Week 2:** focus on analyzing data from USGS. Creating an API request to access the data. Creating my own custom features and comparing them to determine which to use in the model. One way to analyze the dataset is through graphs. I have struggled in the past to understand my datasets fully and to access datasets from past projects. Confirming I can access and analyze the data will give me an advantage over my past projects. By the end of week 2, I will be able to access USGS data and identify which features I want to use in my model. I can't access USGS data, so I will need to find a different dataset. I'm confident USGS data will work.

**Week 3:** Create the base logistic regression model with a proper train/validate/test split. Set up MLflow to hold the model’s accuracy. If I have time, start Hyperparameter tuning by testing out the XGBoost model. Having a whole week working on the model will give me more time if any bugs pop up. By Friday, MLflow will be set up, and the base model accuracy and other metrics will be documented. Not having my MLflow or my model ready will put me behind schedule, making me rush, which could mean data leaks and biases slip past me.

**Week 4:** Create the XGBoost model and hyperparameter-tune it to obtain the best model version. Compare the models and pick the best one for the system. Implementing Airflow to automate the project’s system. This will allow everything to run automatically, so the only thing I need to do is start Next.js. By the end of week four, I will know which model I want to use for my dashboard, and the whole system will be automated, including accessing and cleaning the data and making a prediction. Not having a final model picked would push me back almost a whole week, because I can’t automate the system until the model is picked out and ready to run.

**Week 5:** Create FastAPI and Next.js. On Next.js, take my time to understand what is going on and how to make everything look nice. Prepare for the final presentation. By the end of this week, my project dashboard will be complete, with the automated pipeline running end-to-end. My project would also be ready to be presented to the whole class. I need a dashboard to present on the last day of class; otherwise, I will not complete the assignment requirements.

## Claude Usage Plan

I plan to use Claude as a coding assistant to help me write and understand code. It will be a tutor who helps guide me in my decisions about how I want my system to look and behave. Claude would also be a reviewer, helping me think through all my decisions and informing me when I have made a mistake and how to fix it. Some tasks Claude performs include explaining unfamiliar libraries, debugging, and adding explanations to code comments. More details on what Claude can and can’t do can be found in the Claude.md file at the main root of the project.

My role is to choose the model architecture design. I guide Claude on what I want to happen and why. Any documents written, such as Claude.md, were written by me. I have all final project decisions, so Claude can’t make changes or add anything unless I ask Claude to. Even then, he will get my permission before changing any files/scripts. Some tasks I will handle manually are writing evaluation analyses, making final decisions on anything and everything, and interpreting the results of the different models.
