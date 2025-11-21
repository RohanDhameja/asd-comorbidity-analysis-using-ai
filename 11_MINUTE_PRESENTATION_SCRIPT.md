# 🎤 11-MINUTE PRESENTATION SCRIPT
## Quantifying the Influence of SDOH on Comorbidities in Autistic Children

**Total Time: 11 minutes**  
**Slides: 22**  
**Strategy: Focus on Slides 1-2, 5, 7, 11, 18-20. Skim the rest.**

---

## ⏰ TIMING BREAKDOWN

| Section | Slides | Time | Strategy |
|---------|--------|------|----------|
| **Introduction** | 1-5 | 2 min | Hook them! |
| **Methodology** | 6-8 | 1.5 min | Quick overview |
| **EDA (Skip most)** | 9-14 | 1 min | Skim fast |
| **Statistics** | 15-17 | 1 min | Hit highlights |
| **MODELS (KEY)** | 18-20 | 4 min | ⭐ MAIN FOCUS |
| **Conclusion** | 21-22 | 1 min | Strong close |
| **Q&A Buffer** | - | 0.5 min | Leave room |

---

## 📝 SLIDE-BY-SLIDE SCRIPT

### **SLIDE 1: Title** ⏱️ *0:00-0:20 (20 sec)*

**SAY:**
> "Good morning. I'm Rohan Dhameja from Bellarmine College Prep. Today I'm presenting research on quantifying how socioeconomic factors influence comorbidities in autistic children—using explainable AI."

**ACTION:** Advance immediately, don't pause.

---

### **SLIDE 2: What Are We Talking About?** ⏱️ *0:20-1:00 (40 sec)*

**SAY:**
> "This image shows the problem: Two children with autism, Maya and Sofia. Maya's family has high SES—college-educated parents, quick access to specialists. Sofia's family faces barriers—less education, long wait times. The scales are imbalanced before care even begins. **This creates vastly different health outcomes.**"

**EMPHASIZE:** Point to the unbalanced scales visual.

---

### **SLIDE 3: Global Stats** ⏱️ *1:00-1:20 (20 sec)*

**SAY:**
> "In the U.S., 2.8% of children have autism. But here's the critical number: **70-95% have comorbidities**—ADHD, anxiety, depression, epilepsy. That's nearly universal."

**SKIP:** Don't read all the percentages. Move fast.

---

### **SLIDE 4: Why This Matters (Iceberg)** ⏱️ *1:20-1:40 (20 sec)*

**SAY:**
> "These comorbidities are just the tip of the iceberg. Below the waterline are the hidden forces—poverty, education gaps, healthcare access. **We need to measure these hidden factors.** That's what this research does."

**EMPHASIZE:** Gesture to show "above" vs "below" the waterline.

---

### **SLIDE 5: Research Focus** ⏱️ *1:40-2:00 (20 sec)*

**SAY:**
> "Our five goals: Address disparities, predict risk, inform interventions, characterize social factors, and—most importantly—leverage explainable AI so our results aren't a black box."

**SKIP:** Don't explain each arrow. They can read it.

---

### **SLIDES 6-7: Methodology** ⏱️ *2:00-3:00 (60 sec total)*

**SLIDE 6** (30 sec):
**SAY:**
> "Our methodology: We used the National Survey of Children's Health—2,568 ASD-positive children. We preprocessed the data, trained three machine learning models, and used SHAP analysis for explainability."

**ADVANCE TO SLIDE 7** (30 sec):
**SAY:**
> "More detail for those interested: Label encoding for categorical data, standard scaling, then multi-output prediction to simultaneously predict all four comorbidities. This efficiency was key."

**SKIP:** Don't explain every box. Keep moving.

---

### **SLIDE 8: Dataset** ⏱️ *3:00-3:30 (30 sec)*

**SAY:**
> "Our dataset: 79,000 total children, 2,568 with autism. We used 11 predictor variables—age, gender, but crucially, socioeconomic factors like parental education, income, insurance status. We predicted four binary outcomes: ADHD, anxiety, depression, epilepsy."

**POINT TO:** The table briefly.

---

### **SLIDES 9-14: EDA (SKIM FAST!)** ⏱️ *3:30-4:30 (60 sec total)*

**SAY (while clicking through):**
> "Quick exploratory analysis: Age range 3-17, evenly distributed. [CLICK] Family income follows a normal curve. [CLICK] Here's the key finding: comorbidity prevalence is extremely high—94% ADHD, 84% anxiety, 77% depression. [CLICK, CLICK, CLICK] Correlation heatmap shows weak linear relationships—that's why we need AI. [CLICK, CLICK] Chi-square tests confirmed some significant associations, particularly parental education and depression."

**STRATEGY:** Keep hand on clicker. Advance every 10 seconds. Don't stop.

---

### **SLIDES 15-17: Chi-Square** ⏱️ *4:30-5:00 (30 sec)*

**SAY:**
> "We used Chi-square tests to check for statistical significance. [CLICK] Results: parental education significantly associated with depression, p-value 0.0047. SES linked to ADHD. But simple statistics aren't enough—we need AI to quantify exact influence."

**SKIP:** Don't explain the formula. Just show the p-values matter.

---

### **SLIDE 18: Used Models** ⏱️ *5:00-5:45 (45 sec)* 🌟

**SAY:**
> "We tested three models. **Logistic Regression** for interpretability—it tells us direct odds. **Random Forest** to handle complexity and rank feature importance. **XGBoost** for maximum predictive accuracy. We compared all three to ensure robust findings."

**EMPHASIZE:** This shows scientific rigor.

---

### **SLIDE 19: Classification Report** ⏱️ *5:45-6:45 (60 sec)* 🌟🌟

**SAY:**
> "Results: All models performed well, but **Logistic Regression was optimal**. Look at this: **100% recall**—we didn't miss a single case. Precision 81%, F1-score 0.90. Most importantly, **lowest Hamming loss**—fewest prediction errors. This is the model we used for final analysis."

**EMPHASIZE:** Point to the 1.0 recall bar. This is impressive!

**PAUSE:** Let this sink in for 2 seconds.

---

### **SLIDE 20: SHAP Analysis** ⏱️ *6:45-8:45 (120 sec)* 🌟🌟🌟 **STAR SLIDE**

**SAY:**
> "Now the breakthrough: SHAP analysis. This is explainable AI—it tells us exactly which factors drive predictions.
> 
> [POINT TO EPILEPSY] For epilepsy: Blue dots are low income/education pushing predictions higher. Red dots are high income/education pushing lower. Clear pattern: **low socioeconomic status increases epilepsy risk.**
>
> [POINT TO DEPRESSION] Depression shows the strongest pattern. Low parental education—blue dots—strongly predicts depression. Education is the dominant factor.
>
> [POINT TO ADHD/ANXIETY] ADHD and anxiety show more distributed patterns—these are influenced by multiple factors, not just one.
>
> **The key insight: We can now quantify exactly how much education and income matter. For epilepsy and depression, they're major drivers. This isn't correlation—this is precise, measurable influence from our AI model.**"

**STRATEGY:** 
- Spend 2 full minutes here
- Use your hands to point
- Make eye contact
- This is your money slide!

**IF RUNNING LONG:** Skip ADHD/anxiety details. Focus on epilepsy + depression only.

---

### **SLIDE 21: Limitations** ⏱️ *8:45-9:15 (30 sec)*

**SAY:**
> "Acknowledging limitations: This is cross-sectional data—one time point, so we can't establish causation. It's self-reported, which introduces potential bias. And we lack biological variables like genetics. These limitations guide our future work."

**TONE:** Honest but confident. Shows intellectual maturity.

---

### **SLIDE 22: Future Directions** ⏱️ *9:15-10:00 (45 sec)*

**SAY:**
> "Future directions: First, longitudinal studies to track children over time and establish causation. Second, integrate biological factors—genetics, environmental exposures. Third, advanced modeling with deep learning. Fourth, enhance explainability further. Fifth, cross-cultural validation in other countries. And finally—**policy simulation**: using our model to predict which interventions would actually work before implementing them. That's the ultimate goal—actionable policy change."

**EMPHASIZE:** Point to "Policy Simulation"—this is the real-world impact.

---

### **CLOSING** ⏱️ *10:00-10:30 (30 sec)*

**SAY:**
> "To summarize: We've proven that socioeconomic factors—specifically parental education and family income—are major, quantifiable drivers of comorbidity risk in autistic children. Using explainable AI, we've moved from observation to measurement. This research provides the data necessary to create equitable healthcare policies that level the playing field for every child, regardless of their background.
>
> **Thank you. I'm happy to take questions.**"

**ACTION:** 
- Make eye contact
- Smile
- Stand confidently
- Wait for questions

---

## 🎯 **EMERGENCY TIME SAVERS**

### **If Running OVER Time (at 9 min mark):**

**Cut These Slides Entirely:**
- Slides 9-10 (Age/Income distributions) - "Quick EDA—skip to key findings"
- Slides 12-14 (Correlation details) - "Correlations were weak—that's why we needed AI"
- Slide 16 (Chi-square results table) - Just mention "statistically significant"
- Slide 17 (Comorbidity count) - Skip entirely

### **If Running UNDER Time (at 9 min mark):**

**Add These Points:**
- Mention your dashboard: "I've also built an interactive dashboard where you can explore this data in real-time"
- Expand on SHAP: Explain one more comorbidity in detail
- Add personal motivation: "This research matters because..."

---

## 💡 **PRO TIPS**

### **Voice & Pacing:**
- ✅ Speak slightly faster than normal (but clearly)
- ✅ Pause for 2 seconds after Slide 19 (100% recall)
- ✅ Slow down for Slide 20 (SHAP)—this is key
- ✅ End with confidence on Slide 22

### **Body Language:**
- ✅ Stand to the side of screen, not in front
- ✅ Point to specific parts of charts (Slide 20!)
- ✅ Make eye contact during conclusion
- ✅ Use hand gestures for emphasis

### **Technical:**
- ✅ Have clicker ready
- ✅ Test all animations beforehand
- ✅ Know which slides to advance quickly (9-14)
- ✅ Have water nearby

### **Mental Checkpoints:**
**At 5 min:** Should be on Slide 17-18  
**At 8 min:** Should be on Slide 20 (SHAP)  
**At 10 min:** Should be wrapping up Slide 22  

---

## 🎬 **ONE-SENTENCE SUMMARY PER SLIDE**

**Quick Reference (memorize this):**

1. Title intro - 20 sec
2. **The problem** - scales visual - 40 sec ⭐
3. Stats show universality - 20 sec
4. **Hidden factors** - iceberg - 20 sec ⭐
5. Five research goals - 20 sec
6. Methodology overview - 30 sec
7. Detailed pipeline - 30 sec
8. Dataset specs - 30 sec
9-14. **EDA - CLICK FAST** - 60 sec total 🏃
15-17. Chi-square significance - 30 sec
18. **Three models tested** - 45 sec ⭐
19. **100% recall!** - 60 sec ⭐⭐
20. **SHAP shows education/income drive risk** - 120 sec ⭐⭐⭐
21. Honest limitations - 30 sec
22. **Future: policy simulation** - 45 sec ⭐
Close. Questions - 30 sec

---

## ✅ **FINAL PRE-PRESENTATION CHECKLIST**

### **5 Minutes Before:**
- [ ] Bathroom break
- [ ] Water bottle filled
- [ ] Phone on silent
- [ ] Clicker tested
- [ ] First 3 slides memorized
- [ ] Slide 20 (SHAP) talking points reviewed

### **1 Minute Before:**
- [ ] Deep breath
- [ ] Smile
- [ ] Stand confidently
- [ ] Remember: You know this better than anyone in the room

---

## 🏆 **YOU'VE GOT THIS!**

**Key Success Factors:**
1. **Slides 1-2**: Hook them with the problem
2. **Slides 9-17**: Move FAST (don't get bogged down)
3. **Slides 18-20**: This is your showcase—spend 4 minutes here
4. **Slide 20 (SHAP)**: Your star moment—be confident!
5. **Conclusion**: End strong with policy impact

**Remember:**
- You have 11 minutes—stick to it
- Slides 20 is where you win
- Your research is solid
- You know more than anyone in the audience
- Confidence is key!

**Break a leg! 🎤🧠🚀**

---

**Time Check Right Now:**  
- Read this script out loud
- Time yourself
- Adjust as needed
- You've got this!


