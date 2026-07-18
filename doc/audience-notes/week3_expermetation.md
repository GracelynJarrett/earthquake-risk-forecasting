# Porter
**Summary**
    Porter did a lot of model experimenting, finding that having the emotional model helps increase the model's metrics. Porter also tested removing the content of the scenes. He also played around with the position in the scene. In the end, Porter picked the model that was most honest over the model with the best metrics.
**Presentation Strengths**
    Showed clear metrics and was able to explain them.
**Improvements**
    Needs to explain new vocabulary and new models as he goes, not just when someone asks him to explain it. Needs to work on time management — I feel like the last couple of weeks Porter's presentations are always the longest.
**Question (I) Asked**
    Last week there were 3 models, this week there are 4 — how many models are being used?
    **Answer:** MiniLM, BiLSTM, and positional are all layers; the emotional model is a separate model. (This is how I understand Porter's response.)

**Quick notes**
- The model is not finding the turning points.
- The model is not doing well for the point of no return (for the movie Die Hart).
- Every model that sees position ties a model that sees nothing.
- The model is trying to guess the turning point without looking at the right context.
- No input: the content in the scene gets thrown out; only the title of the scene gets put in.
- Transforming position: the scenes have all content, and are in order.
- Transforming no position: the scenes have all content, but are in a different order.
- Went with the model that was the most honest.
- Chosen model: positional_prior.
- positional_prior looks at the scenes around the turning point (new model).
- Next week's plan is to use synopsis -> scene retrieval.
- MiniLM, BiLSTM, and positional are all layers; the emotional model is separate.
- Low accuracy is ok (32 and 41).
- Model accuracy increased when the emotional model was added.
- Right now: train on silver, test on gold.
- Try combining and randomly picking.




# Quinn
**Summary**
    Through experimentation, Quinn was able to find that the more data he gives the model, the better DICE metric he can get. Quinn also found that his model is really good at identifying the pancreas, but not good at identifying the tumor/sick part. Quinn is also using specificity, which shows the model's trust.
**Presentation Strengths**
- Had an example dashboard to fully show us how his model works. Was able to explain how and why he changed the trimming of the images.
**Improvements**
- Time management.
**Question (I) Asked**
- Did not ask a question.
**Quick notes**
- Used 9-view — medical uses.
- Only trains on the model.
- The bias is trimming the images — but it has proven to increase probability.
- CT column (images) ->SegResNet (pretrained SuPreM) -> output.
- Most important metric is the DICE (outlines the pancreas).
- Specificity went from 8 -> 55.
- DICE [accuracy], specificity [trust].
- Did not work: loss reweighting, bigger field of view, balanced patch sampling, finer resolution (helps the pancreas, not the tumor).
- Did work: MORE DATA, whole-box ROI.
- 1,412 cases, DICE 53%.
- Detecting the pancreas is easier than detecting cancer.
- Hulls are evidence, data is the lever.
- Images are now 128x128.
- Box-hole is the same size as 1 pixel, but in 3D space.



# Ted
**Summary**
    Ted's project has changed since the beginning. He went from 8 hand signs down to 2, but now his model can track his hand instead of having a box for the user to keep their hand in. Ted plans to add one more strategy to his project: if the hand sign is not a fist, then consider it a palm. That way, if the user is between hand signs, it will always fall back to palm.
**Presentation Strengths**
    Having a small demonstration prepared to show how his webcam works.
**Improvements**
    I don't know.
**Question (I) Asked**
- Does lighting have any effect?
    **Answer:** Has not seen any effect; the model gets confused when the hand is tilted.
- Does speed have an effect?
    **Answer:** The model does not pick up slow movement, but it can pick up fast movement.

**Quick notes**
- Webcam frame -> MediaPipe Hand Landmarker -> MobileNetV3 head -> cursor + click.
- Three different strategies: 1. full-frame single-stage (53%), 2. detect -> crop -> classify (92%), 3. cursor + binary click (98%).
- Changed from 8 hand signs down to 2.
- Hand detected 48%, classify 71%, ???? 34%.
- Add one more strategy: if not fisted (sign 1), then it's palm (sign 2).




# Me
**Summary**
    Overall, I think I had a good presentation. I was able to get through my whole slide show within the 10-minute time slot, and I was able to explain everything that I wanted to. Based on the feedback I received, I need to do more experimentation around what features to have and why. I also need to rethink my original proposal, because besides creating the features and splitting into test/validate/train, I don't use the timestamp anymore. This means the data leakage that I originally proposed and was trying to stop probably was unnecessary. I am also not using a time-series forecasting model, so changing the split, validation, and train shouldn't cause any leakage.
**Presentation Strengths**
    I had good time management while presenting. When answering questions, though, my time management wasn't as good. I felt like I was able to show all of the points I wanted to share.
**Improvements**
    Need to improve on answering audience questions — I stumble a lot and had a hard time explaining some stuff. I also need to work on my pronunciation of my words.
**Quick notes**
- Look at feature importances.
- See what would happen if we mixed up the splits.
- heck feature significant 