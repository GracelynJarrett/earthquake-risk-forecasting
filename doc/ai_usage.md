# AI Usage Log
**Project:** Earthquake Risk Forecasting
A weekly record of project progress and how AI tools were used each week.

-------
## Week X — [Date]

**What we worked on:** Brief summary of the week's focus and topics.
**Deliverables completed:** What was actually built, written, or committed this week.
**Feedback received:** Any instructor feedback or direction.
**Action items:** What to prioritize or do differently next week.
**Reflection:** One or two sentences — anything surprising or worth thinking about.

### AI Usage
- **Tasks:** What you used AI assistance for
- **Prompts that worked well:** Specific prompts or context that were effective
- **Corrections needed:** Any cases where AI output needed fixing or redirection
------

## Week 1 — June 30–July 3, 2026

**What we worked on:** Project setup and the Week 1 proposal deliverables. Chose the project (predicting the likelihood of a large earthquake, magnitude 4.5+, in the next week for California, Japan, and Greece), set up the GitHub repo and folder structure, and wrote the proposal, schedule, Claude.md, and this AI usage log. Also prepared and gave the Thursday pitch.

**Deliverables completed:** GitHub repo + folder structure; Claude.md (moved to repo root); proposal.md; schedule.md; ai_usage.md; pitch + audience notes.

**Feedback received:** Instructor recommended SQLite instead of CSV for data storage. Pitch feedback: consider switching Greece for another Ring of Fire location; decide between one model for all three regions or three separate models (and how to avoid overfitting); narrow the audience further (e.g., hospitals) and clarify what they'd do with the info; address user trust given past earthquake models "crying wolf."

**Action items:** Begin Week 2 data work — pull USGS data for all three regions, analyze, clean, and load into SQLite. Decide the one-vs-three-model question. Weigh the region and audience feedback.

**Reflection:** Overall, I feel like this week went really well. I was able to do some research into other models that did similar things and how my model is going to be different. I also received a lot of great feedback from my pitch about what I should think about moving forward.

### AI Usage
- **Tasks:** Formatting the proposal (plain text → headings + Tech Stack table); proofreading the proposal, schedule, and pitch notes; reviewing proposal content (flagged the missing magnitude threshold, the missing storage/database entry, and an Airflow week mismatch); building the schedule skeleton and drafting Week 3–5 day plans; clarifying and editing the Claude.md guidelines.
- **Prompts that worked well:** Clear, specific asks like *"fix the formatting — Tech Stack should be a table"* and *"fix spelling and grammar in week1_pitch.md"*; and asking Claude *"tell me if you don't understand something"* up front, which surfaced useful clarifying questions before any work started.
- **Corrections needed:** Early on, Claude tried to write the entire proposal in one shot, which broke my rule of working one section at a time with me owning the writing. I stopped it and pointed to Claude.md; after that it proposed changes and waited for approval. I also had it undo/adjust things when I switched storage from CSV to SQLite.


