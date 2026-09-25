# Codebook for the rater comments (Table 5)

Unit: one free-text comment. A comment can carry several themes or none. Code what the comment says, not what the coder thinks of the item. The first coding was made by one author with access to the labels and item statistics (Section 3.5). A second author reviews every code with the item shown.

## Weak distractors (`distractors`)

Table 5: One or more distractors implausible, contradictory or excludable without domain knowledge.

The comment says that at least one wrong option (B, C or D) is nonsensical, silly, strange or implausible, contradicts the stem, or can be ruled out without knowing the machine-learning topic.

Examples:

- "D nonsensical"
- "B nonsensical."

## Depends on recall or teaching (`recall_dep`)

Table 5: Level depends on what was taught; key repeats a statement from the material.

The comment says that the answer hinges on remembering a specific statement, definition or example from the course material (for instance, the key repeats the text), or that the Bloom level depends on how the topic was taught.

Examples:

- "A is basically the definition of SSL (remember)"
- "Depends on context, create if the student has not seen this solved."

## Common sense suffices (`common_sense`)

Table 5: Answerable by reading comprehension or common sense.

The comment says that the item can be answered by careful reading, general knowledge or common sense, without any machine-learning knowledge.

Examples:

- "B-D can be excluded without much knowledge/effort."
- "B-D somewhat silly. Understand in the sense of reading comprehension / common sense."

## Flawed stem or key (`flaw`)

Table 5: Stem or key ambiguous or flawed.

The comment says that the question or the correct answer is ambiguous, wrong, badly worded or confusing, or that another option could also be correct. A general complaint about the quality of the item counts.

Examples:

- "Horrible question!"
- "C-D non-plausible. Isn't B a plausible answer, in addition to A?"

## Level capped by weak alternatives (`lower_bound`)

Table 5: Level cannot be evaluate/analyze because the alternatives are indefensible.

The comment argues explicitly that the item cannot be at a high level such as evaluate or analyze, because the alternatives are not defensible choices that a student would have to weigh.

Examples:

- "Analyze type of question but B-D nonsensical."
- "B-D are not plausible which means that "evaluate" doesn't apply."

## Key gives itself away (`giveaway`)

Table 5: Wording of the key gives the answer away.

The comment says that the wording of the correct answer reveals it, for example because it repeats words from the stem or stands out from the other options.

Examples:

- "Text in A sort of gives away the answer, B-D are nonsensical."
- "B-D do not have much to with data contracts and silent failures, also the text in A sort of gives away the answer."

## Rater unfamiliar with topic (`unfamiliar`)

Table 5: Rater unfamiliar with the topic.

The rater says that they are not familiar with the topic of the item.

Examples:

- "I'm not familiar with this topic but B-D can be excluded since they are silly."
- "I'm not familiar with this topic. Bloom level prediction suffers (benefits perhaps?)"
