# Stakeholder Notes — Earthquake Risk Forecasting

Working notes on **who** the weekly regional forecast is for and **how** it should be used.
Living document — add to it as thinking develops. (Feeds the presentation Q&A and the Week 5 business layer.)

---

## 1. Who is the stakeholder? (narrow to ONE primary)
The forecast is a **weekly, per-region risk level**. Which decision-maker acts on that?

**My primary stakeholder:** Fire departments / first responders

**Why them:** This project began with the Utah wildfires, which got me thinking
about what other natural disasters communities must prepare for. Fire crews are
exactly the teams who adjust readiness ahead of a disaster, so a weekly regional
risk level maps directly to their staffing and equipment decisions.
---

## 2. How can stakeholders trust us?
- **No data leakage** — honest metrics (the whole point of this project; other models inflate accuracy by cheating)
- **Validated on unseen future data** — strict temporal train/validate/test split
- **Report probabilities, not false certainty** — communicate a risk level with its uncertainty
- **Be explicit about limits** — predicts *regional* risk for the next 7 days, NOT the exact time, city, or magnitude
- **Show a track record over time** — let stakeholders see calibration (when we say 20% risk, it happens ~20% of the time)

---

## 3. What should stakeholders DO with a prediction?
- **Low risk week**
  - Do reguler dutes
  - stay alert
  - be prepared for anything
    - have proper equiment/ vehicles avable
    - have proper medical supplys

- ** Elevated risk week**
  - More crew members on call
  - Off duty members be ready to come in if needed
  - pre-check equipment/ vehicles
  - review responcse routs
    - routs up today
    - be aware of any obsticals like constrution
    - where schools, stores, and hospitase are located

- **High risk week**
  - Have a plain for if people lose there homes
    - where to house them
    - have enofe food
    - familys stay together
  - Cleaning up high destution area
    - area around old buileds are clear just incase they fall
    - Gas lines secared properly
    - water lines secared properly
  - Make suer smaller deparments have enofe supplys and cre members
  - Make suer mebical supplys are up todate

---

## 4. Gray scenarios (edge cases to think through)
- **False alarm** (high risk, nothing happens) → cost of over-preparing; "alarm fatigue" if it happens often
- **Missed event** (low risk, big quake strikes) → the dangerous case; must be clear the model is *not* a guarantee
- **Borderline risk (~50%)** → how should a stakeholder act on an in-between number?
- **"High" means different things per region** → base rates differ (CA 15% vs Japan 40%); contextualize what "elevated" means locally
- **Data lag / catalog artifacts** → a live prediction depends on timely, complete data; the 2009-style reporting gaps show data can shift

Notes: _______________

---

## 5. Spatial granularity — region vs. city
- The model predicts at the **region level** (California / Japan / Greece), NOT a specific city
- Predicting the exact city is **out of scope**, not feasible with this approach, and would **over-promise** (hurts trust)
- We DO store each quake's latitude/longitude, so we *could* show **where recent activity has clustered** as context — without claiming to predict the city
- Possible **future** direction, not current scope

Notes: _______________



## Links
  - https://www.sciencedirect.com/science/article/abs/pii/S0379711221001387
  