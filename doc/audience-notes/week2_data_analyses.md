
# Ted
**Summary**
    Ted has two different model options he is deciding between for his project. The first option uses two different models: the first model would detect and isolate the hand sign, then put it through the pipeline, which would crop the image, then crop it a second time to just the hand sign. The second model would only be trained on the image that is just the hand sign. The second option is having one model but multiplying the images by rotating, flipping, and playing around with different hyperparameters. In the end Ted did a great job at explaining his project and answering questions.
**Presentation Strengths**
    Was prepared to answer any question. Was honest when he was not sure about something.
**Improvements**
    I don't know.
**Question (I) Asked**
 Are the different hand signs in different positions (front vs side view), and would left and right hand affect anything?
    Response: Yes, there are images from left and right hands; unsure if there are different hand positions (front vs side view).
 Would skin color have any effect on the model?
    Response: At this time don't know, but Ted is concerned about it too.

**Quick notes**
    - 2 different data [annotated JSON file] [image]
    - using 8 out of 18 hand movements
    - Option one: Have one model find the hand sign, another model identify the hand sign
        - the pipeline has three different steps: 1. recognizing where the hand sign is, 2. crop image, 3. zoom into the hand sign
    - Option two: rotating/flipping the images throughout the pipeline process (no cropping, or v-box) [prepared model]
    - the size of the hand image depends on how far the player will be away from the computer (table, desk)
    - recommendation from audience: feed all three images: 1. the raw, 2. background noise cropped out, 3. the close-up hand sign
    - have 15 frames per image (how long do the users have to hold the hand sign)
    - Have both left and right hand, one hand position (thumbs up from the side, don't know if have images from the front [showing knuckles])
    - only change from pitch is not sure which model option to go with at this point (Ted is leaning towards option 2)
    - Don't know if skin color and the person would have an effect on the model


# Quinn
**Summary**
    Quinn has analyzed his dataset and found that his dataset is made up of many smaller datasets; however, since Hopkins medical school standardized all the data, they all have the same columns. The only difference is that some images are the whole torso and some are a smaller area. The test set has more tumors than the train set; this is on purpose to make it easier to validate the model. Quinn has made one model at this point and he compared it to Hopkins' model's metrics. One of Quinn's metrics is close to the Hopkins model. Quinn did a great job of explaining why he made the decisions that he did and how he plans to use more than just the images for his capstone if he decides to make this his capstone project.
**Presentation Strengths**
    Demonstrated that he knows his data, and why it's organized the way that it is.
**Improvements**
    Gotten better but still could improve on time management.
**Question (I) Asked**
    Did not ask a question
**Quick notes**
    - two data sets
    - images are more important than the other columns (FOR THIS PROJECT)
    - the train set has 9.8% tumors, test has 16.8%, so it's easy to validate
    - different size in images, different clarity due to different hospitals
    - dataset is compiled from different hospitals
    - there are some images falsely identified but easy to see the difference
    - using RAM to hold the images
    - takes a long time to train the model
    - outline has not changed
    - Similar to Ted but 3D not box
    - Using Dice overlap
    - Lesion is the tumor (all tumors are lesions, all lesions are NOT tumors)
    - recommendation for audience: smaller dataset
    - all mini datasets were standardized to all have the same columns



# Porter
**Summary**
    Porter has a dataset from TRIPOD that is marked by many different people; these people marked down when major turning points were in the script. Porter wants to put the dataset into a pre-trained screen reader (why, I am not sure), then use those results and put them into a neural network that would identify different TPs. It is unclear as to why he needs the screen reader model. Overall it feels like Porter's project is very complex with a lot of moving parts, and right now he is lost and is struggling to tell us what he is trying to say.
**Presentation Strengths**
    Had very nice slides, received feedback with open arms.
**Improvements**
    Struggled to explain what the screen reader did and why it was in the pipeline. Went over the time limit by a lot.
**Question (I) Asked**
    Didn't ask a question
**Quick notes**
    - right now have TRIPOD: screenplays, gold label, silver label, scriptBase
    - training on the silver, testing on the gold
    - Looks at TP-TP5 (these are scenes where a situation event happens)
    - Had some data quality issues in silver
    - Some screenplays had different labels, some had non-standard sluglines
    - opportunity, change of plans, point of no return, major setback, climax [TP1-TP5]
    - takes raw data -> pre-trained AI scene reader (miniLM) -> TP finder (small neural net: Porter-trained [neural network]) -> TP outputs
    - Scene turns when the emotion changes. (Using an emotion AI model "how positive or negative does this feel")
    - Plans changed a lot (got caught up with trying to find and understand any and everything that could go in the model)
    - Tried to go full white box, but changed to going black box
    - switch from statistical model to deep learning model
    - miniLM will read through the script and find what scene the turning point is at
    - audience recommendation: combine and mix silver and gold


# ME
**Summary**
    Overall I feel like I did really well on this presentation. I had all of my graphs ready to go, and was able to use my graphs to answer questions anyone had asked me. I received some feedback on looking into the seasonal and monthly significance, and seeing if they would add any value to my model. Right now I have graphs from all of my data pulled; I need to look at the timeline for just above my threshold. However, I did feel like I was talking really fast — I was afraid I would go over the time limit so I tried to go a little faster, but in the end I finished before the time went off.
**Presentation Strengths**
    I was prepared to answer anyone's questions and I was able to answer some of their questions with the graphs.
**Improvements**
    I felt like I talked too fast and was scrolling through all of my graphs a lot; maybe next time isolate what graphs I want to share in their own file so I am not scrolling everywhere.
**Quick notes**
 - If have time, look at three different models, one per region
 - look at threshold and above to see if month should be a feature
