# Daily Log
**Project:** Earthquake Risk Forecasting
Short daily notes on what I worked on, what's next, and any blockers.

-----------
## [Week X — Day] — [Date]

**Worked on:** What did you actually work on today?
**Up next:** What are you working on in the next session?
**Blockers:** Are you facing any challenges or blockers? If none, write "None."
------------
# Week 1
## [Week 1 - 1] June 29, 2026
First day of class — the project was introduced.

## [Week 1 - 2] June 30, 2026
**Worked on:** Today I set up my GitHub, created my main folder, and added md files into the doc folder. I created the Claude.md file and started to work on the proposal.md.
**Up next:** Tomorrow I will work on finishing my proposal and my schedule. I will also need to write my pitch for Thursday.
**Blockers:** None

## [Week 1 - 3] July 1, 2026
**Worked on:** Finished writing the proposal and schedule. I did more research into why other models are the way they are. I worked on my pitch outline.
**Up next:** Finish pitch outline and present it to the class.
**Blockers:** I keep second-guessing myself.

**What we discussed:** We discussed how my model would be different than the others, why the other models were not the best, how I can improve my pitch for tomorrow, and finding my "why" for doing this project.
**Feedback received:** Need to do more research. Why do the other models have data leakage? What features are those models using? Why would data leakage be bad? How can I make this project unique from other models? Good topic to show what I have learned through the year. Need to find the "why" — why are you doing this project?
**Action items:** Need to do more research on the other models — see why their models had data leaks, what features they were using, and what will make my project unique from the others.
**Reflection:** I have a good topic and it would work for this project, but I could push myself to do a project that puts me out of my comfort zone. I need to do more research as to why and how my model will be different than other models.

## [Week 1 - 4] July 2, 2026
**Worked on:** Finished writing my pitch, and finished reading the study about why and how other models failed at predicting earthquakes. Presented the pitch to the class.
**Up next:** Waiting for final approval of the project.
**Blockers:** I was nervous about the presentation.

------------------------------------------------------
# week 2
## [Week 2 - 1] July 6, 2026
**Worked on:** Today I was able to create a script to test pulling the records worldwide in the last 24 hours, where I got a result of 45 earthquakes. I also created a historical script that pulled data from each region starting in 2000 to the present year. I had time and started to set up the Jupyter notebook for tomorrow.
**Up next:** Start analyzing the data, find what features I am thinking about using and why, and compare the different data sets. I might even pull one more region of New Zealand.
**Blockers:** No blockers for today.

**NOTES** - California has 4x more earthquakes than the other regions, probably due to its size and that USGS is a USA organization.
    - Japan has an outlier of a 9.7 earthquake in 2011 (this was expected).
    - Japan's lowest recorded earthquake is 2.7, not 2.0, due to Japan not recording/reporting anything lower.


## [Week 2 - 2] July 7, 2026
**Worked on:** Today I was able to analyze the data and narrow down what features I know I want to have and why. The only feature I am unsure about is the depth of the earthquake. There is no big correlation between the depth and magnitude, but I am curious to see the model's accuracy and other metrics with and without the depth feature.
**Up next:** Creating the data cleaning pipeline.
**Blockers:** I had trouble getting the kernel for my Jupyter notebook to work, but I eventually fixed it.


## [Week 2 - 3] July 8, 2026
**Worked on:** Today I was able to make the cleaning pipeline, added in the distance-from-fault feature, and saved all the clean data into a database.
**Up next:** Tomorrow I will be working on all the documentation. I will start with declaring what features I want to use and why, and what my target variable is.
**Blockers:** No blockers for today.


## [Week 2 - 4] July 9, 2026 (unofficial check-in)
**Worked on:** Today I got all of my documentation done. I created the implementation plan and data understanding report, and figured out what I want to present for tomorrow. I updated my schedule, adding more detail to what I will be doing each day and when I know I have reached my goal. The schedule file is now designed as a to-do list. I also added a stakeholder note file for me to write down notes about my stakeholder, where I narrowed down from emergency responders to firefighters.
**Up next:**
**Blockers:**


## [Week 2 - 5] July 10, 2026
**Worked on:** Today was presentation day. I think I did really well, and I received some feedback to look more into the seasonal and monthly timeline for just my data over my threshold.
**Up next:** Over the weekend, look into the timeline over my threshold. Starting Monday, I will set up my MLflow and start working on integrating the metrics that I need to gather live, then process.
**Blockers:** I have work Saturday and Monday, and I also need to work on my capstone proposal.

----------------------------------------------------------------------------------
# week3
## [Week 3 - 1] July 13, 2026
**Worked on:** Today I set up MLflow, did more analysis on whether I should add month as a feature (the answer was no), and I updated some documents (requirements.txt). I also split the data into its train, test, and validation sets. I was even able to make the build-feature script, which means I am a day ahead and should have more time to run more base model tests.
**Up next:** Tomorrow I will create my first base model and slowly add features in (depth, lat/lon, fault-line) to find the best model.
**Blockers:** No blockers for today.


## [Week 3 - 2] July 14, 2026
**Worked on:** Today was very productive. I was able to train and validate 20 different models. 15 of the 20 models were looking at training the models individually. However, from a quick glance, the very first base model with all of the tested features did the best.
**Up next:** Tomorrow I will dive deeper into analyzing the different models to clarify which base model I would like to use.
**Blockers:** No blockers for today.


## [Week 3 - 3] July 15, 2026
**Worked on:** Today I worked on finalizing my model experimentation, and started my experimentation report.
**Up next:** Tomorrow I will finish the report and create my slides for my slide show presentation.
**Blockers:** No blockers for today.

**What we discussed:**
    We talked about how I have many experimental models and how they all have low metrics, which was expected. We talked about doing more experiments with taking out the temporal features to see if the model would improve. We also talked about finding a way to make this project more trustworthy for users, and not just crying wolf.
**Feedback received:**
    Still need to find a way to show users that my model is safe to use, and that I should look into adding more temporal features and removing the temporal features.
**Action items:**
    Look into removing the temporal features.
    Look into adding temporal features over years 1, 3, 5, + in magnitude.
    Look into pulling in explosion data to see any correlation.
    Look into how the current model's features correspond to the model.
    Look into precision, not just AUC score.
**Reflection:**
    Overall, I think I am at a good spot. I agree that I should look into what my model would look like if there were no temporal features. I am still trying to find a way to guarantee that users can trust my model and that it's not just crying wolf. I am also still doing some research into what firefighters would do if they had time to prepare for an earthquake, and what would happen if my model had false positives and false negatives.



## [Week 3 - 4] July 16, 2026
**Worked on:** Today I worked on creating my PowerPoint for tomorrow.
**Up next:** Making my PowerPoint look nice and then presenting to the class.
**Blockers:** Today I stayed and watched all of the Capstone presentations, and didn't get a lot of time to work on my project.


## [Week 3 - 5] July 17, 2026
**Worked on:** Today I finished my PowerPoint and presented. I think overall I did an ok job at presenting. I was able to receive feedback, and now I have a lot of other things I can experiment with.
**Up next:** Later today, and this weekend, I will finish writing my report, finish the capstone assessment, and get ahead in other classes so next week I can focus on working on this project.
**Blockers:** I have work.

--------------------------------------------------------------------------------------------------------------
# Week 4
## [Week 4 - 1] July 20, 2026
**Worked on:** Today I had a small invervew with cluade asking questions about my project. I updataed my Schegeral adding in the extra suff I want/need to do. I train my first GXBoost model based on the best base model feachers. I also got the feachers importens for GXBoost
**Up next:** Do some more expermantation with adding more feachers and taking away feachers
**Blockers:** I had work and other class work that needs to get done as well


## [Week 4 - 2 ] July 21, 2026 (1 on 1 checkend)
**Worked on:** Today I was able to get a lot done. I did more expermtation with the feachers and was able to get my AUP score up to 37 (i rounded up). I found that the feacher region was not needed along with the feachers that had longer durations, so years in setead of days and some 30d feachers where taken out. I also played around with the metric threshold but in the end I left them how they orginaly was.
**Up next:** Tomorrow I will focuse on testing out changing the Split for a strict time split to a randome split to see what would happan. I also will start my Hypertuning.
**Blockers:** I have homework in other classes that need to be done.

**What we discussed:** Briefly summarize the key topics covered in the check-in.
    Todays chekin was mainly around what I see as my biggest chalage though out all of my AI Classes. For me its being stubern about what I want done and how to get it done. However I have goten better at not bing so stubern and more of going with the flow and seeing what happans. We talked about how much I have improved over the classes and how I can contuine to work on my stubbernces and when being stubberne is ok. 
**Feedback received:** What specific feedback or direction did the instructor give?
    I have improve in my Presintatin skills, showing more confidence and being more prepared to present. Also that Its ok to be stubbernce with some parts of projects but I need to be more opean about my plains changing. Being more flexable in my methodes and plans for my project. 
**Action items:** What will you do differently or prioritize as a result?
    Try to be more opean about having my plans change and how I can handel them. 
**Reflection:** In one or two sentences — was there anything that surprised you or that you want to think more about?
    No nothing surprized me about my chanlige, I have seen know I was stubern about my project plans and how to implment them for a while. I do want to think about different ways I can be more opean to my plans changing, and being more aware when its ok to be stubern and when its ok that plans changed. 




## [Week 4 - 3 ] July 22, 2026
**Worked on:** Today I was able to test out spliting the data set on random. The metrics whent up however it still consiter data leageae, do to recodes having the same vaules around the same time meaning the modle could memerize fake pattrons. I was also able to hypro tune my chosen model and at the test set. My final model now has a AUC metric of 0.40 wich is where I was hoping to have my model. 
**Up next:** Tomorrow I will be having my inverew talk. I will also work on creating my airflow and seting my systom up to be adamadic. 
**Blockers:** I had other homework I neede to work on


## [Week 4 - 4 ] July 23, 2026
**Worked on:** Today we have invervews and I did better then I though I would so I am prode about that. I was able to get Airflow set up.
**Up next:** I will test Airflow all the way though and then start setting up the fastapi end points
**Blockers:** no Blockers


## [Week  - ] July , 2026
**Worked on:**
**Up next:**
**Blockers:**


## [Week  - ] July , 2026
**Worked on:**
**Up next:**
**Blockers:**


## [Week  - ] July , 2026
**Worked on:**
**Up next:**
**Blockers:**


## [Week  - ] July , 2026
**Worked on:**
**Up next:**
**Blockers:**


## [Week  - ] July , 2026
**Worked on:**
**Up next:**
**Blockers:**


## [Week  - ] July , 2026
**Worked on:**
**Up next:**
**Blockers:**


## [Week  - ] July , 2026
**Worked on:**
**Up next:**
**Blockers:**


## [Week  - ] July , 2026
**Worked on:**
**Up next:**
**Blockers:**


## [Week  - ] July , 2026
**Worked on:**
**Up next:**
**Blockers:**