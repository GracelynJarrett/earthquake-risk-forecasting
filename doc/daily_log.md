# Daily Log
**Project:** Earthquake Risk Forecasting
Short daily notes on what I worked on, what's next, and any blockers.

-----------
## [Week X — Day] — [Date]

**Worked on:** What did you actually work on today?
**Up next:** What are you working on in the next session?
**Blockers:** Are you facing any challenges or blockers? If none, write "None."
------------

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

## [Week 2 - 1] July 6, 2026
** Worked on:** Today I was able to creat a scrite to test pulling the records world wide in the last 24 hours; where I got a resulte of 45 earthquakes. I also created a historcal script that pulled data from each reagion staring in 2000 - present year. I had time and started to set up the Jupiter notbook for tomorrow 
**Up Next:** start analyzing the data, find what feachers I am thing about using and why, compare the differnet data sets. I might even pull one more region of Newzelinde
**Blockers"** No Blockers for today. 

**NOTES** - Califona haves 4x more eathquakes then the other regions probaly due to its size and that USGS is a USA orginzation
    - Japan haves an outlyer of 9.7 eatherquake in 2011 (this was expeted)
    - Japans loweset recorded eatherquake is 2.7 not 2.0 do to Japan not recording/reporting anything lower


## [Week 2 - 2] July 7, 2026
**Worked on:** Today I was able to analyze the data and norow down what feaches I know I want to have and why. The only feacher I am unsher about is the depth of the earthquake. There is no big coralation from the depth and magmitude but I am curios to see the models accursy and other metrics with and with out the depth feacher. 
**Up Next:** Creating the data cleaning pipline
**Blockers:** I had truble geting my kunal for my jupiternotebook to work but I avencaly fixed it


## [Week 2 - 3] July 8, 2026
**Worked on:** Today I was able to make the cleaning pipeline, added in the distance-from-fault feature, and saved all the clean data into a database.
**UP Next:** Tomorrow I will be working on all the documentation. I will start with declaring what features I want to use and why, and what my target variable is. 
**Blockers:** No blockers for today


## [Week 2 - 4] July 9, 2026 (unofficial check-in)
**Worked on:** Today I got all of my documentation done. I created the implementation plan and data understanding report, and figured out what I want to present for tomorrow. I updated my schedule, adding more detail to what I will be doing each day and when I know I have reached my goal. The schedule file is now designed as a to-do list.  I also added a stakeholder note file for me to write down notes about my stakeholder, where I narrowed down from emergency responders to firefighters. 
**UP Next:**
**Blockers:**


## [Week 2 - 5 ] July 10, 2026
**Worked on:** Today was presentation day. I think I did really well, and I received some feedback to look more into the seasonal and monthly timeline for just my data over my threshold.
**UP Next:** Over the weekend, look into the timeline over my threshold. Starting Monday, I will set up my MLflow and start working on integrating the metrics that I need to gather live, then process. 
**Blockers:** I have work Saturday and Monday, and I also need to work on my capstone proposal.

----------------------------------------------------------------------------------

## [Week 3 - 1 ] July 13, 2026
**Worked on:** Today I set up mlflow, did more analyses on if I should add month as a feacher (the anwer was no), I updated some documents (requiremnts.txt). I also split the data into its train, test, and validation sets. i was even able to make the build feature script, witch means I am a day ahead, and should have more time to run more base model tests.
**UP Next:** Tomorrow I will creat my first base model and sloly add feachers in (depth, lad/lon, faltline) to find the best model. 
**Blockers:** No Blockers for today



## [Week 3 - 2 ] July 14, 2026
**Worked on:** Today was vary predutive, I was able to train and validate 20 different models. 15 of the 20 mondels where looking at training the models indvicaly. However from a quick glance the vary first Baise model we all of the tested feachers out did the best. 
**UP Next:** Tomorrow I will dive deeper into analyzing the different models to clarify wich base model I would like use.
**Blockers:** No Blockers for today. 


## [Week 3 - 2 ] July 15, 2026
**Worked on:** Today I worked on finlaixing my model expemetation, and started my experimention report.
**UP Next:** Tomorrow I will finnish the report and creat my slides for my slide show presitation.
**Blockers:** No Bloakcers for today

**What we discussed:** 
    we talked about how I have many expermental modles and how they all have a low metrics, wich was expeted. We talked about doing more experments with taking out the temperal feachers to see if the model would inprove. We also talked about finding a way to make this project more trustworthy for users, and not just crying wolf.
**Feedback received:** 
        Still need to find a way to show users that my modle is safe to use, and that I should look into adding more temperal feachers and removing the temeral feachesrs. 
**Action items:**
    Look in to removing the temperal feachers
    look into added temperal feachers in years 1,3, 5, + in magmituded
    Look inot pulling in explotion data to see any coralation
    Look into current models feachers coraspontis to the models
    Look into presigion not just USC score
**Reflection:** 
        Over all I think I am at a good spot, I agree that I should look into what my modle would look like if there was no temperal feachers. I am still trying to find a way to garinty that users can trust my model and that its not just crying wolf. I am also still doing some reacher into what FireFighters would do if they had time to prepar for an earth quack. And what would happan if my model had False positeve and False Negitves. 



## [Week  - ] July , 2026
**Worked on:**
**UP Next:**
**Blockers:**


## [Week  - ] July , 2026
**Worked on:**
**UP Next:**
**Blockers:**


## [Week  - ] July , 2026
**Worked on:**
**UP Next:**
**Blockers:**


## [Week  - ] July , 2026
**Worked on:**
**UP Next:**
**Blockers:**


## [Week  - ] July , 2026
**Worked on:**
**UP Next:**
**Blockers:**


## [Week  - ] July , 2026
**Worked on:**
**UP Next:**
**Blockers:**


## [Week  - ] July , 2026
**Worked on:**
**UP Next:**
**Blockers:**


## [Week  - ] July , 2026
**Worked on:**
**UP Next:**
**Blockers:**


## [Week  - ] July , 2026
**Worked on:**
**UP Next:**
**Blockers:**


## [Week  - ] July , 2026
**Worked on:**
**UP Next:**
**Blockers:**


## [Week  - ] July , 2026
**Worked on:**
**UP Next:**
**Blockers:**


## [Week  - ] July , 2026
**Worked on:**
**UP Next:**
**Blockers:**


## [Week  - ] July , 2026
**Worked on:**
**UP Next:**
**Blockers:**


## [Week  - ] July , 2026
**Worked on:**
**UP Next:**
**Blockers:**


## [Week  - ] July , 2026
**Worked on:**
**UP Next:**
**Blockers:**