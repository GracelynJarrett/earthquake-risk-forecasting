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


## Week 2 — July 6–10, 2026

**What we worked on:** Week 2 data foundation. Pulled USGS earthquake data (magnitude 2.0+, 2000–present) for California, Japan, and Greece via the FDSN API; ran a full EDA; cleaned the data and loaded it into a SQLite database; added a distance-to-fault feature; finalized the prediction target and feature list; and wrote the data-understanding report and supporting docs. Presented the data understanding on Friday.

**Deliverables completed:** Three pipeline scripts (`fetch_usgs.py`, `clean_data.py`, `load_to_sqlite.py`); SQLite database (`earthquakes.db`, 176,376 rows); EDA notebook with graphs, deep-dives, and interpretations; `data-understanding-report.md` (Sections 1–5); `implementation-plan.md`; `stakeholder-notes.md`; `schedule.md` rebuilt as a checklist; `Claude.md` refresh; committed plate-boundary reference data.

**Feedback received:** From the presentation — look into seasonal and monthly significance for data above my threshold. From my professor during the week — add a magnitude vs. distance-to-fault graph and show p-values for all columns.

**Action items:** Over the weekend, look into the timeline above my threshold. Week 3: set up MLflow, build the temporal split, engineer features, train the baseline logistic regression, and run the 5-run ablation. Optionally switch the EDA behavior graphs to the cleaned data for consistency.

**Reflection:** 
After analyzing my dataset, the biggest change I have is my thresholds should be different per region. This makes sense because each region is different. Japan typically has larger earthquakes and California has a lot of smaller ones. 
This week Claude was very helpful in keeping me on track with my original plan. Sometimes I would overthink something or lose sight of my original plan, and Claude would help me get back on track. 

### AI Usage
- **Tasks:** Wrote the pipeline scripts section by section with plain-language explanations; designed and refined the EDA graphs; verified every statistic against the real data before it went into the report; troubleshot a Jupyter kernel that hung (downgraded `ipykernel`); added the distance-to-fault feature with `shapely`; talked through the target definition and feature list; drafted the report via a fact-sheet, plus the schedule, implementation plan, and Claude.md refresh.
- **Prompts that worked well:** Asking Claude to *"verify the numbers against the real data before writing"* (no guessed stats); the fact-sheet → Copilot → Claude accuracy-check workflow for big docs (kept credit use low); asking Claude to explain each code section before writing it.
- **Corrections needed:** Told Claude to keep imports and each graph in their own notebook cells; stopped it from editing my `.ipynb` directly (VSCode overwrote the edits) and switched to it giving me cells to paste; scoped library use (Plotly only for the map + presentation); made the final calls on decisions Claude surfaced (region-specific thresholds; keeping the leakage columns in storage behind a feature allow-list). Claude also fixed Copilot's mangled report tables and strengthened the leakage write-up.


