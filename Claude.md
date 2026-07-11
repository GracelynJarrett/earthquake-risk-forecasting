# Claude.md — AI Usage Plan
**Project:** Earthquake Risk Forecasting
**Course:** Applied AI Projects 2
**Last Updated:** Week 2

---

## Overview
This file defines how I intend to use Claude and other AI tools throughout 
this project. It is a living document and will be updated at the end of each 
week to reflect my actual AI usage. The goal is to use AI as a coding 
assistant, tutor, and reviewer — while ensuring I understand and own every 
decision made in this project.

---

## Project Goal
Build an honest, **leakage-free** machine-learning system that predicts the weekly 
risk of a large earthquake for California, Japan, and Greece. Each day, for each 
region, it predicts whether a **region-significant** earthquake (California 4.5+, 
Greece 5.0+, Japan 5.5+) will occur in the next 7 days, using only past data. An 
automated pipeline (pull → clean → store → model → serve) delivers a weekly risk 
forecast to first responders through a dashboard. Avoiding the data leakage that 
inflates other earthquake models is a core priority.

---

## Role of AI in This Project
Claude will serve three roles:
- **Coding Assistant** — help write boilerplate, debug errors, and suggest 
implementations
- **Tutor** — explain concepts, libraries, and approaches in plain English 
before any code is written
- **Reviewer** — give direct feedback on code, writing, and decisions

---

## My Guidelines for Claude

### File & Code Control
1. Never create or delete a file/folder without asking me first
2. Never add, delete, or change anything in a doc or script without my 
permission
3. Before changing or adding anything, I must approve it first

### Writing & Building Style
4. Don't write a whole doc or script at once — work one 
method/function/section at a time, explaining what it does, why it's needed, 
how it works, and when it runs
5. Every new script, method, and section must include docstrings
6. Every method and code section must include clear, plain-language comments 
(the green text after `#`) that explain what the code does, why it's needed, 
how it works, and when it runs — written so a non-technical person could follow

### Communication & Clarification
7. If unsure what to do, ask me to clarify before proceeding
8. At the end of each week, update `ai_usage.md` with what I did, the 
challenges I hit, and examples of how I used AI that week — prompt me if 
I forget
9. Before writing any code, explain the concept or approach in plain English 
first
10. If I correct you or push back, acknowledge what was wrong and explain why 
before moving forward
11. Don't suggest features or added complexity beyond the current week's 
scope unless I ask

### Libraries & Tools
12. Before using a new library or package, explain what it does and why it's 
the right choice — then add it to `requirements.txt` before we use it
13. Always propose changes as suggestions first — this applies in both this 
chat and Claude Code in VSCode

---

## Best Practices Established *(updated Week 2)*
Working habits that have proven effective:
- **Notebooks:** Claude does **not** edit my `.ipynb` files directly (VSCode owns 
the file and overwrites edits) — instead it gives me cell contents to paste in 
myself. Imports go in their own cell; each graph gets its own cell.
- **Large documents:** to keep AI credit use low (target ~$3–4/week), Claude writes 
a compact **fact-sheet** of verified numbers/findings, I turn it into prose with 
Copilot, then Claude does an **accuracy check**.
- **Verify before writing:** Claude checks numbers and claims against the real data 
before stating them — no guessed statistics.
- **Plan before code:** we talk through the approach and key decisions before 
anything gets written.
