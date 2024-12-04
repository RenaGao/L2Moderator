
attributes = {
    "dialogue act": {
        "type": "multiclass",
        "options": {
            "Information Probing":{
                "definition":"Encourages participants to share their thoughts, opinions, or experiences by posing questions or directly inviting input from individuals or the group.",
                "examples": '“Anyone else who want to share their thought or opinion about why they what is the purpose of relationship for them?” “Related to stress. And how do you manage in such situation?” “Yeah. How about you, Chantelle?” “Anyone else who want to share their thought or opinion about why they what is the purpose of relationship for them?” “Do you agree with this statement?”'
            },
            "Clarification Requests":{
                "definition":"Seek clarification from participants to ensure understanding or resolve ambiguities in their contributions.",
                "examples": '"What is that? Do you mind explaining it?" "Do you mean the people are not honest?" "Could you explain again what you just said?" "It is not suitable for laughing, but you want to laugh?" "What do you mean by cultural differences in this context?"'
            },
            "Opinion Sharing": {
                "definition":"Express personal views, beliefs, or subjective interpretations related to the discussion topic.",
                "examples": '"Maybe it. You know, it is not even about the weekend and how you spend it." "And if you get used to the comfort level of AI, I am not sure how human how people can get along with other humans." "For me, managing stress is all about maintaining a good work-life balance." "To me, sharing housework equally is a sign of respect and partnership in a relationship." "I believe that having diverse perspectives in a team leads to more creative solutions."'
            },
            "Information Sharing": {
                "definition":"Clarify, reframe, summarize, paraphrase, or make connection to earlier conversation content.",
                "examples": '"Today is topic is related to the recent trends in the job market." "Internal friction refers to when someone feels mentally trapped by their worries or thoughts." "In some countries, housework is traditionally divided along gender lines, which can lead to conflicts." "Research shows that group discussions can improve second language acquisition by increasing practice opportunities." "The concept of burnout was first defined in the 1970s as a state of emotional and physical exhaustion." "You can find free online courses on platforms like Coursera or edX to learn new skills." "The unemployment rate has decreased by 2% in the last quarter, according to recent statistics."'
            },
            "Echoing": {
                "definition":"Reinforce or relate to a prior statement by sharing similar experiences, ideas, or observations.",
                "examples": '"Yeah, my friend had a similar experience when he was in the US, as he struggled to find a job." "I had a similar idea, that uses AI to facilitate communication." "I can relate to that feeling of being overwhelmed when learning a new language. I have been through it too." "That reminds me of a conversation I had with my mentor; they said almost the same thing about career growth." "I heard something similar from a colleague about how AI is being integrated into education." "I completely agree—my friend also struggled with finding a balance between work and family responsibilities."'
            },
            "Personal Story Sharing": {
                "definition":"Share a specific, personal experience or anecdote to illustrate a point, connect with others, or contribute to the discussion.",
                "examples": '"He is a businessman. He just signed a contract, so I was very stressful, like Whoa!" "There was a time when I had to make a tough decision about changing my career path—it was such a challenging moment for me." "Once, during my university days, I stayed up all night preparing for a group project because I wanted everything to be perfect."'
            },
            "Emotional Expression": {
                "definition": "Articulate emotions or emotional reactions in response to the discussion, the topic, or another participant’s experience.",
                "examples": '"The type, the the time feel much, much faster, just like, you know, because our relationships are already stabilized, and there were nothing really." "I feel sorry for the people who work seven days a week." "I am glad you talk about this point." "I am sorry for the accident and the bad time you have gone through." "I’m really excited about this idea—it has so much potential!" "I feel frustrated when I think about how unfair the system can be sometimes." "It makes me so happy to see how everyone is participating in this discussion." "I feel a bit nervous sharing my thoughts, but I think it’s important to speak up." "I am truly inspired by the example you just gave—it’s really motivating."'
            },
            "Acknowledgement": {
                "definition":"Recognize, validate, or show appreciation for another participant’s contribution, insight, or effort during the discussion.",
                "examples": '"Okay, yeah, I think that is a very interesting discussion." "That is a very interesting insight." "And so I think Urania has made some very interesting and also very important points." "Yeah, absolutely. I feel like even just showing the willingness to do or share housework deserves applause." "Great point, and I think it really ties back to what we were discussing earlier." "I appreciate you bringing this up—it’s a really valuable perspective." "You’re absolutely right. That’s something I hadn’t considered before." "Thanks for sharing that example—it really helped clarify the idea."'
            },
            "Backchanneling": {
                "definition":"Brief verbal or non-verbal responses for indicating active listening, understanding, or agreement during a conversation.",
                "examples": '"Yeah." "Hmm." "Okay." "Uh-huh." "Right." "I see." "Oh." "Mhm."'
            },
            "Courtesy Expressions": {
                "definition":"Use polite or respectful phrases to facilitate smooth and considerate communication.",
                "examples": '"Goodbye!" "Thank you!" "Please, go ahead!" "Excuse me." "I appreciate your time."'
            },
            "Interpretation": {
                "definition":"Interpretate, clarify, reframe, summarize, paraphrase, or make connection to earlier conversation content.",
                "examples": '"If I understand correctly, you’re suggesting that online courses are beneficial because they provide flexibility." "This idea connects to what we discussed earlier about the importance of time management in high-pressure jobs." "To summarize, the main takeaway here is that building relationships in the workplace helps reduce stress." "What you’re describing reminds me of the earlier example about AI being used to facilitate team discussions." "In other words, you’re arguing that peer feedback plays a critical role in language learning success." "This ties back to the point John made earlier about how small gestures can have a big impact on teamwork."'
            },
            "Coordinative Instruction": {
                "definition":"Explicitly command, influence, or halt the immediate behavior of the recipients for coordinating the process and structure of the session.",
                "examples": '"Can we wrap up this discussion and move on to the next point?", "I’d like everyone to think about this question and share your thoughts one by one.", "Now everyone is here, lets start the session.", "Please turn off your microphone when you are not speaking."'
            },
        }
    },

}


def construct_prompt_unit(instance):
    prompt = ""

    topic = instance["meta"]["topic"]
    speakers = instance["meta"]["speakers"]
    target = instance["target"]
    instruction = f'Your role is an annotator, annotating the moderation behavior of a second language speakers" English conversation session. The topic is "{topic}", given the definition and the examples, the context of prior and posterior dialogue, '

    instruction += "please label which dialogue act the target sentence belong to? And who is the moderator talking to?"

    instruction +=  '\n\n'
    prompt += instruction

    speakers = ", ".join([f'"{s}"' for s in speakers])


    dialogue_act_intro = "Dialogue act: Dialogue acts is referring to the function of a piece of a speech/sentence. The definitions and examples of the dialogue acts are below:\n"
    prompt += dialogue_act_intro

    dialogue_acts = attributes["dialogue act"]["options"]
    index = 0
    for name, info in dialogue_acts.items():
        act_def = f'{index}. {name}: {info["definition"]} \n'
        act_ex = f'examples: {info["examples"]} \n\n'
        prompt += act_def
        prompt += act_ex
        index += 1

    if len(instance["context"]["prior_context"]) > 0:
        prompt += "Dialogue context before the target sentence:\n"
        for s in instance["context"]["prior_context"]:
            prompt += f"{s[0]} ({s[1]}): {s[2]} \n"

    prompt += "\nTarget sentence:\n"
    prompt += f"{target[0]} ({target[1]}): {target[2]} \n"

    if len(instance["context"]["post_context"]) > 0:
        prompt += "\nDialogue context after the target sentence:\n"
        for s in instance["context"]["post_context"]:
            prompt += f"{s[0]} ({s[1]}): {s[2]} \n"

    prompt += "\n"

    dialogue_acts_string = ", ".join([f"{i} ({name})" for i, (name, info) in enumerate(dialogue_acts.items())])
    prompt += "Please answer only for the target sentence with the JSON format:{"
    prompt += f'"dialogue act": String(one option from {dialogue_acts_string}),'
    prompt += '"target speaker(s)": String(one option from ' + speakers + '),'
    prompt += '"reason": String'
    prompt += "}\n"
    prompt += "For example: \n"
    prompt += 'answer: {"dialogue act": "1 (Information Probing)", "target speaker(s)": "3 (Joe Smith)", "reason": "The moderator asks a question to Joe Smith aimed at eliciting his viewpoint or reaction to a statement from the recent policy change for combatting climate change......"}'

    return prompt