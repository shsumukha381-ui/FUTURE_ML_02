# How the Support Ticket Classifier Works

*A plain-language guide for non-technical stakeholders*

---

## What Does This System Do?

This system reads support ticket text and automatically:
1. **Categorizes** the ticket into one of 8 IT service categories
2. **Assigns a priority level** (High, Medium, or Low)
3. **Provides confidence scores** showing how certain the model is about each prediction

Think of it as an experienced support agent who instantly triages incoming tickets — except it never gets tired, works 24/7, and processes tickets in milliseconds.

---

## How Does Categorization Work?

### The Intuition

Imagine you're a support manager who has read thousands of tickets. Over time, you develop an instinct: when you see words like "laptop", "screen", "broken", "replacement", you immediately think "Hardware." When you see "password", "login", "access denied", you think "Access."

The machine learning model does exactly this, but mathematically.

### The Process

1. **Text Cleaning**: The raw ticket text is cleaned up — lowercased, stripped of punctuation, and common filler words like "the", "is", "and" are removed. This focuses on the meaningful words.

2. **Word Importance Scoring (TF-IDF)**: Each word gets a score based on two factors:
   - **How often** it appears in this particular ticket (Term Frequency)
   - **How rare** it is across all tickets (Inverse Document Frequency)

   Words that appear frequently in one ticket but rarely in others are the most informative. For example, "laptop" is very informative for Hardware tickets but rare in HR tickets.

3. **Pattern Matching**: The model (a mathematical formula trained on thousands of real tickets) looks at the word scores and calculates which category pattern the ticket most closely matches. It also considers two-word phrases (like "access denied" or "hard drive") for better accuracy.

4. **Confidence**: The model doesn't just give a single answer — it provides a probability for each category. If it says "Hardware: 87%, Access: 8%, Storage: 3%...", it's quite confident the ticket is about Hardware.

### The Categories

| Category | What It Covers |
|---|---|
| **Hardware** | Laptop, monitor, keyboard, mouse, printer issues, hardware replacements |
| **HR Support** | Employee onboarding, offboarding, policy questions, benefits |
| **Access** | Login problems, password resets, account access, VPN issues |
| **Miscellaneous** | General inquiries that don't fit other categories |
| **Storage** | File storage, shared drives, backup, data space requests |
| **Purchase** | Equipment procurement, software licenses, budget approvals |
| **Internal Project** | Project-related requests, resource allocation, planning |
| **Administrative rights** | Permission changes, elevated access, admin privileges |

---

## How Is Priority Decided?

### The Challenge

The original dataset did not include priority labels. Instead of making up fake priorities, we built a transparent, rule-based system to derive priority, then trained a model to learn these patterns.

### The Priority Rules

Priority is determined by combining three signals:

#### 1. Category Urgency (Primary Signal)
Some categories are inherently more urgent than others:

| Base Priority | Categories | Reasoning |
|---|---|---|
| **High** | Access, Administrative rights | Blocks employee from working |
| **Medium** | Hardware, Storage | Impacts productivity but may have workarounds |
| **Low** | HR Support, Purchase, Internal Project, Miscellaneous | Generally planned or non-blocking |

#### 2. Urgency Keywords (Modifier)
Certain words signal that a ticket needs faster attention:
- **Strong urgency**: "urgent", "asap", "critical", "emergency", "immediately"
- **Blocking issues**: "blocked", "down", "crashed", "broken", "not working"
- **Security concerns**: "unauthorized", "breach", "locked", "expired"

If one urgency keyword is found, priority is bumped up by one level.
If two or more are found, it's bumped up by two levels.

#### 3. Complexity (Modifier)
Very long tickets (top 10% by word count) tend to describe complex, escalated issues. These get bumped up by one level.

### Example

> *"My laptop screen is broken and I need a replacement urgently."*

- Category: **Hardware** → Base priority: Medium (1)
- Keyword: **"urgently"** → +1 modifier
- Result: **High priority**

### Why This Approach?

This rule-based derivation is:
- **Transparent**: You can see exactly why a ticket got a certain priority
- **Adjustable**: Change the rules in the config if your team's priorities differ
- **Honest**: We clearly document that priority is derived, not from ground truth

The ML model then learns these patterns from the text alone, so in production it can predict priority without running the rules — just from the words in the ticket.

---

## How Good Is the Model?

The model is evaluated on a held-out test set (20% of tickets not used during training) using standard metrics:

- **Accuracy**: What percentage of predictions are correct overall
- **Precision**: Of tickets predicted as category X, what % actually are X
- **Recall**: Of tickets that are actually X, what % did we correctly identify
- **F1 Score**: The harmonic mean of precision and recall (balances both)

We report both **macro** (equally weight all classes) and **weighted** (weight by class size) averages, since some categories have far more tickets than others.

Detailed results with confusion matrices and per-class breakdowns are available in the `reports/` directory.

---

## Key Limitations

1. **Priority is derived, not ground truth**: The priority model is only as good as the derivation rules. If your team's actual prioritization logic differs, the rules in `config.py` should be adjusted.

2. **Domain-specific**: The model is trained on IT support tickets. It may not perform well on customer support tickets from other domains (e.g., e-commerce, SaaS) without retraining.

3. **Pre-processed data**: The training data appears to have been partially pre-processed (stopwords removed, lowercased). The cleaning pipeline accounts for this.

4. **Confidence ≠ correctness**: A high confidence score means the model is certain about its prediction, not that it's necessarily right. Always verify edge cases.

---

## Glossary

| Term | Meaning |
|---|---|
| **TF-IDF** | A way to score words by importance — frequent in this ticket but rare overall = important |
| **Cross-validation** | Testing the model multiple times on different data splits to ensure it's not just memorizing |
| **Macro F1** | Average F1 across all classes, treating each class equally regardless of size |
| **Weighted F1** | Average F1 weighted by how many tickets each class has |
| **Confusion matrix** | A table showing what the model predicted vs. what was actually correct |
