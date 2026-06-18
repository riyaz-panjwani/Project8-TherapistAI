"""Prepare joint intent + DST training data for TherapistTransformer.

~25 examples per class → ~325 total → ~276 train / 49 val.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

SEED_DATA: list[dict] = [
    # ── crisis ──────────────────────────────────────────────────────────────
    {"text": "I've been thinking about hurting myself",         "intent": "crisis"},
    {"text": "I don't want to be here anymore",                 "intent": "crisis"},
    {"text": "I've been having thoughts of suicide",            "intent": "crisis"},
    {"text": "I've been cutting myself again",                  "intent": "crisis"},
    {"text": "I want to end it all",                            "intent": "crisis"},
    {"text": "I keep thinking about not waking up",             "intent": "crisis"},
    {"text": "I feel like disappearing forever",                "intent": "crisis"},
    {"text": "I've been hurting myself to cope",                "intent": "crisis"},
    {"text": "Nobody would miss me if I was gone",              "intent": "crisis"},
    {"text": "I planned how I would do it",                     "intent": "crisis"},
    {"text": "I took too many pills last night",                "intent": "crisis"},
    {"text": "I feel like everyone would be better without me", "intent": "crisis"},
    {"text": "I can't see a reason to keep going",              "intent": "crisis"},
    {"text": "I've been writing goodbye notes",                 "intent": "crisis"},
    {"text": "The pain is unbearable I just want it to stop",   "intent": "crisis"},
    {"text": "I have a plan to kill myself",                    "intent": "crisis"},
    {"text": "I scratched my arms until they bled",             "intent": "crisis"},
    {"text": "I feel completely hopeless and want to die",      "intent": "crisis"},
    {"text": "I'm scared of what I might do to myself tonight", "intent": "crisis"},
    {"text": "Life feels not worth living anymore",             "intent": "crisis"},
    {"text": "I've been giving away my things",                 "intent": "crisis"},
    {"text": "I stood at the edge of the bridge for a long time", "intent": "crisis"},
    {"text": "I can't stop thinking about suicide",             "intent": "crisis"},
    {"text": "I burned myself again",                           "intent": "crisis"},
    {"text": "I really don't want to be alive",                 "intent": "crisis"},

    # ── anxiety ─────────────────────────────────────────────────────────────
    {"text": "I can't stop worrying about everything",          "intent": "anxiety"},
    {"text": "I had a panic attack on the tube this morning",   "intent": "anxiety"},
    {"text": "My heart races every time I think about it",      "intent": "anxiety"},
    {"text": "I feel overwhelmed all the time",                 "intent": "anxiety"},
    {"text": "I keep catastrophising about the future",         "intent": "anxiety"},
    {"text": "I feel so anxious and I can't breathe",           "intent": "anxiety"},
    {"text": "I get dizzy and shaky when I'm stressed",         "intent": "anxiety"},
    {"text": "My mind won't stop racing at night",              "intent": "anxiety"},
    {"text": "I feel nervous for no reason all the time",       "intent": "anxiety"},
    {"text": "I avoid going outside because of my anxiety",     "intent": "anxiety"},
    {"text": "I'm terrified of making mistakes at work",        "intent": "anxiety"},
    {"text": "I had a panic attack in the middle of class",     "intent": "anxiety"},
    {"text": "I worry constantly that something bad will happen","intent": "anxiety"},
    {"text": "Even small decisions make me incredibly anxious", "intent": "anxiety"},
    {"text": "I feel like something awful is about to happen",  "intent": "anxiety"},
    {"text": "My hands shake when I'm in social situations",    "intent": "anxiety"},
    {"text": "I can't sleep because my brain won't stop",       "intent": "anxiety"},
    {"text": "I feel a tight knot in my chest constantly",      "intent": "anxiety"},
    {"text": "I'm scared to leave the house some days",         "intent": "anxiety"},
    {"text": "The dread I feel every morning is exhausting",    "intent": "anxiety"},
    {"text": "I overthink every conversation I have",           "intent": "anxiety"},
    {"text": "I've been having really bad health anxiety",      "intent": "anxiety"},
    {"text": "I cancelled plans again because of my nerves",    "intent": "anxiety"},
    {"text": "I feel paralysed by all my worries",              "intent": "anxiety"},
    {"text": "I can feel my anxiety getting worse",             "intent": "anxiety"},

    # ── depression ──────────────────────────────────────────────────────────
    {"text": "I just feel empty like nothing matters",          "intent": "depression"},
    {"text": "I can't get out of bed most mornings",            "intent": "depression"},
    {"text": "Everything feels pointless",                      "intent": "depression"},
    {"text": "I've lost interest in things I used to love",     "intent": "depression"},
    {"text": "I feel hopeless and numb",                        "intent": "depression"},
    {"text": "I'm exhausted all the time for no reason",        "intent": "depression"},
    {"text": "I don't feel anything anymore",                   "intent": "depression"},
    {"text": "Getting through the day feels impossible",        "intent": "depression"},
    {"text": "I haven't showered or eaten properly in days",    "intent": "depression"},
    {"text": "I feel like a burden to everyone around me",      "intent": "depression"},
    {"text": "There's this heavy darkness I can't shake",       "intent": "depression"},
    {"text": "I stopped enjoying the things I used to love",    "intent": "depression"},
    {"text": "I feel completely disconnected from everything",  "intent": "depression"},
    {"text": "I cry for no reason most days",                   "intent": "depression"},
    {"text": "I feel like I'm just going through the motions",  "intent": "depression"},
    {"text": "I don't see the point in anything anymore",       "intent": "depression"},
    {"text": "I've been isolating myself from everyone",        "intent": "depression"},
    {"text": "My depression has been really bad lately",        "intent": "depression"},
    {"text": "I feel worthless and I don't know why",           "intent": "depression"},
    {"text": "I can't concentrate on anything",                 "intent": "depression"},
    {"text": "I feel like nothing will ever get better",        "intent": "depression"},
    {"text": "I just want to sleep and not wake up for a while","intent": "depression"},
    {"text": "I don't recognise myself anymore",                "intent": "depression"},
    {"text": "The sadness won't go away no matter what I do",   "intent": "depression"},
    {"text": "I feel so low today I could barely move",         "intent": "depression"},

    # ── venting ─────────────────────────────────────────────────────────────
    {"text": "I'm so sick of how my flatmate treats me",        "intent": "venting"},
    {"text": "My boss is an absolute nightmare and I'm furious","intent": "venting"},
    {"text": "I just need to vent today was terrible",          "intent": "venting"},
    {"text": "I'm fed up with everything",                      "intent": "venting"},
    {"text": "I'm so pissed off right now",                     "intent": "venting"},
    {"text": "I can't stand this anymore",                      "intent": "venting"},
    {"text": "Everything went wrong today and I'm so angry",    "intent": "venting"},
    {"text": "Nobody listens to me and it's driving me mad",    "intent": "venting"},
    {"text": "I'm so frustrated I could scream",                "intent": "venting"},
    {"text": "Today was an absolute disaster from start to finish","intent": "venting"},
    {"text": "I hate everything about my situation right now",  "intent": "venting"},
    {"text": "I just need to get this off my chest",            "intent": "venting"},
    {"text": "I'm beyond exhausted with all of it",             "intent": "venting"},
    {"text": "I snapped at someone today and I feel awful",     "intent": "venting"},
    {"text": "Nothing is going right and I'm so angry",         "intent": "venting"},
    {"text": "I'm sick and tired of being taken for granted",   "intent": "venting"},
    {"text": "The day was woke up pissed nobody respected me",  "intent": "venting"},
    {"text": "I missed my bus and everything after just fell apart","intent": "venting"},
    {"text": "I had the worst commute and I'm still fuming",    "intent": "venting"},
    {"text": "I'm annoyed and I just need someone to hear me",  "intent": "venting"},
    {"text": "I kept getting interrupted in every meeting",     "intent": "venting"},
    {"text": "Everyone ignored my ideas again and I'm livid",   "intent": "venting"},
    {"text": "I got soaked in the rain and missed the train",   "intent": "venting"},
    {"text": "I got blamed for something that wasn't my fault", "intent": "venting"},
    {"text": "I'm just really frustrated today",                "intent": "venting"},

    # ── seeking_advice ──────────────────────────────────────────────────────
    {"text": "I don't know what to do about my relationship",   "intent": "seeking_advice"},
    {"text": "Should I quit my job",                            "intent": "seeking_advice"},
    {"text": "What do you think I should do",                   "intent": "seeking_advice"},
    {"text": "How do I deal with this",                         "intent": "seeking_advice"},
    {"text": "Any advice on how to handle this",                "intent": "seeking_advice"},
    {"text": "I'm not sure whether to confront them or let it go","intent": "seeking_advice"},
    {"text": "What would you do if you were in my position",    "intent": "seeking_advice"},
    {"text": "I need help figuring out what to do next",        "intent": "seeking_advice"},
    {"text": "Should I tell them how I feel",                   "intent": "seeking_advice"},
    {"text": "I need some guidance on this situation",          "intent": "seeking_advice"},
    {"text": "Can you help me think through this decision",     "intent": "seeking_advice"},
    {"text": "I have no idea how to approach this",             "intent": "seeking_advice"},
    {"text": "What's the right thing to do here",               "intent": "seeking_advice"},
    {"text": "I'm torn between two options and don't know which","intent": "seeking_advice"},
    {"text": "How do I tell my parents about this",             "intent": "seeking_advice"},
    {"text": "Is it okay to set boundaries with family",        "intent": "seeking_advice"},
    {"text": "What should I say to my partner about this",      "intent": "seeking_advice"},
    {"text": "I need advice on how to stop procrastinating",    "intent": "seeking_advice"},
    {"text": "How can I stop feeling guilty all the time",      "intent": "seeking_advice"},
    {"text": "Should I see a therapist or try to manage alone", "intent": "seeking_advice"},
    {"text": "How do I stop comparing myself to everyone",      "intent": "seeking_advice"},
    {"text": "What do I do when I feel like giving up",         "intent": "seeking_advice"},
    {"text": "How do I start setting boundaries",               "intent": "seeking_advice"},
    {"text": "I want to get better but don't know how",         "intent": "seeking_advice"},
    {"text": "What steps can I take to manage my anxiety",      "intent": "seeking_advice"},

    # ── relationship ────────────────────────────────────────────────────────
    {"text": "My partner and I keep having the same argument",  "intent": "relationship"},
    {"text": "I feel so lonely even when I'm with people",      "intent": "relationship"},
    {"text": "My mum doesn't understand me at all",             "intent": "relationship"},
    {"text": "I had a huge fight with my boyfriend",            "intent": "relationship"},
    {"text": "My friend betrayed my trust",                     "intent": "relationship"},
    {"text": "I feel disconnected from my family",              "intent": "relationship"},
    {"text": "My girlfriend and I haven't been getting along",  "intent": "relationship"},
    {"text": "I feel invisible in my friendships",              "intent": "relationship"},
    {"text": "My dad said something that really hurt me",       "intent": "relationship"},
    {"text": "I don't feel close to anyone anymore",            "intent": "relationship"},
    {"text": "My best friend and I had a falling out",          "intent": "relationship"},
    {"text": "I feel like my partner doesn't appreciate me",    "intent": "relationship"},
    {"text": "My sister keeps criticising everything I do",     "intent": "relationship"},
    {"text": "I feel completely alone even surrounded by people","intent": "relationship"},
    {"text": "My parents don't support my choices",             "intent": "relationship"},
    {"text": "My relationship feels one sided",                 "intent": "relationship"},
    {"text": "I push people away and I don't know why",         "intent": "relationship"},
    {"text": "I'm scared of being abandoned",                   "intent": "relationship"},
    {"text": "My colleague keeps undermining me",               "intent": "relationship"},
    {"text": "I miss the closeness we used to have",            "intent": "relationship"},
    {"text": "I don't have anyone I can really talk to",        "intent": "relationship"},
    {"text": "My family doesn't know about my mental health",   "intent": "relationship"},
    {"text": "I feel like a burden to the people I love",       "intent": "relationship"},
    {"text": "My partner doesn't take my emotions seriously",   "intent": "relationship"},
    {"text": "I struggle to trust people after being hurt",     "intent": "relationship"},

    # ── work_stress ─────────────────────────────────────────────────────────
    {"text": "I have three deadlines this week and I can't cope","intent": "work_stress"},
    {"text": "I think I'm burning out from work",               "intent": "work_stress"},
    {"text": "My exams are in two weeks and I'm not ready",     "intent": "work_stress"},
    {"text": "My boss keeps piling on work",                    "intent": "work_stress"},
    {"text": "I stayed at the office until midnight again",     "intent": "work_stress"},
    {"text": "I'm dreading going into work tomorrow",           "intent": "work_stress"},
    {"text": "I haven't had a day off in three weeks",          "intent": "work_stress"},
    {"text": "My workload is completely unmanageable",          "intent": "work_stress"},
    {"text": "I got a bad performance review and I'm devastated","intent": "work_stress"},
    {"text": "I'm terrified of failing my dissertation",        "intent": "work_stress"},
    {"text": "I feel like I'm always behind at work",           "intent": "work_stress"},
    {"text": "I got shouted at by my manager in front of everyone","intent": "work_stress"},
    {"text": "University is overwhelming me",                   "intent": "work_stress"},
    {"text": "I feel like I'm going to get fired",              "intent": "work_stress"},
    {"text": "I missed a deadline and I feel terrible",         "intent": "work_stress"},
    {"text": "I work from home and I can't switch off",         "intent": "work_stress"},
    {"text": "I'm struggling to keep up with my studies",       "intent": "work_stress"},
    {"text": "I feel like a failure at my job",                 "intent": "work_stress"},
    {"text": "My commute is exhausting and adds two hours",     "intent": "work_stress"},
    {"text": "I was passed over for promotion again",           "intent": "work_stress"},
    {"text": "I'm stressed about money because work is slow",   "intent": "work_stress"},
    {"text": "I have an important presentation tomorrow",       "intent": "work_stress"},
    {"text": "I got made redundant and I don't know what to do","intent": "work_stress"},
    {"text": "I'm struggling to balance uni and a part time job","intent": "work_stress"},
    {"text": "I feel completely underwater at work",            "intent": "work_stress"},

    # ── self_esteem ─────────────────────────────────────────────────────────
    {"text": "I hate how I look",                               "intent": "self_esteem"},
    {"text": "I'm not smart enough for this",                   "intent": "self_esteem"},
    {"text": "Everyone else seems to have their life together", "intent": "self_esteem"},
    {"text": "I feel like I'm not good enough",                 "intent": "self_esteem"},
    {"text": "I keep comparing myself to others",               "intent": "self_esteem"},
    {"text": "I don't think I deserve to be happy",             "intent": "self_esteem"},
    {"text": "I feel ugly and worthless",                       "intent": "self_esteem"},
    {"text": "I'm embarrassed by who I am",                     "intent": "self_esteem"},
    {"text": "I feel stupid compared to everyone around me",    "intent": "self_esteem"},
    {"text": "I can never do anything right",                   "intent": "self_esteem"},
    {"text": "I'm my own worst critic",                         "intent": "self_esteem"},
    {"text": "I hate myself sometimes",                         "intent": "self_esteem"},
    {"text": "I feel like a fraud and people will find out",    "intent": "self_esteem"},
    {"text": "I look in the mirror and feel disgusted",         "intent": "self_esteem"},
    {"text": "I'm not as successful as I should be at my age", "intent": "self_esteem"},
    {"text": "I don't think I'm likeable",                      "intent": "self_esteem"},
    {"text": "I feel ashamed of my past",                       "intent": "self_esteem"},
    {"text": "I can't accept compliments",                      "intent": "self_esteem"},
    {"text": "I feel inferior to everyone",                     "intent": "self_esteem"},
    {"text": "I put myself down constantly",                    "intent": "self_esteem"},
    {"text": "I'm terrified of being judged",                   "intent": "self_esteem"},
    {"text": "I feel unlovable",                                "intent": "self_esteem"},
    {"text": "I never feel like I'm enough",                    "intent": "self_esteem"},
    {"text": "I feel so insecure all the time",                 "intent": "self_esteem"},
    {"text": "I'm scared of failing so I don't even try",       "intent": "self_esteem"},

    # ── trauma ──────────────────────────────────────────────────────────────
    {"text": "Something happened to me as a child I've never talked about","intent": "trauma"},
    {"text": "I keep having flashbacks from what happened",     "intent": "trauma"},
    {"text": "I was abused and I still can't get past it",      "intent": "trauma"},
    {"text": "I have nightmares about what happened",           "intent": "trauma"},
    {"text": "Certain things trigger memories I'd rather forget","intent": "trauma"},
    {"text": "I feel unsafe even when there's no real danger",  "intent": "trauma"},
    {"text": "My body freezes when I'm reminded of the past",   "intent": "trauma"},
    {"text": "I've never told anyone what really happened",     "intent": "trauma"},
    {"text": "I still feel shame about something that was done to me","intent": "trauma"},
    {"text": "I can't talk about it without breaking down",     "intent": "trauma"},
    {"text": "The past keeps coming back even when I try to forget","intent": "trauma"},
    {"text": "I was assaulted and I still don't feel safe",     "intent": "trauma"},
    {"text": "I've been carrying this secret for years",        "intent": "trauma"},
    {"text": "I startle at loud noises ever since it happened", "intent": "trauma"},
    {"text": "I feel dirty because of what was done to me",     "intent": "trauma"},
    {"text": "I dissociate sometimes when things remind me of it","intent": "trauma"},
    {"text": "I was in a really bad accident and I'm not over it","intent": "trauma"},
    {"text": "Certain smells take me right back to what happened","intent": "trauma"},
    {"text": "I've been diagnosed with PTSD",                   "intent": "trauma"},
    {"text": "My childhood was not safe",                       "intent": "trauma"},
    {"text": "I was emotionally abused for years",              "intent": "trauma"},
    {"text": "I blame myself for what happened",                "intent": "trauma"},
    {"text": "I never feel fully present because of the past",  "intent": "trauma"},
    {"text": "I saw something terrible that I can't unsee",     "intent": "trauma"},
    {"text": "I have PTSD from my relationship",                "intent": "trauma"},

    # ── gratitude ───────────────────────────────────────────────────────────
    {"text": "Talking to you really helped me last time",       "intent": "gratitude"},
    {"text": "Thank you I feel so much better",                 "intent": "gratitude"},
    {"text": "I really appreciate you listening",               "intent": "gratitude"},
    {"text": "This has been so helpful thank you",              "intent": "gratitude"},
    {"text": "I feel heard and that means a lot",               "intent": "gratitude"},
    {"text": "You really helped me see things differently",     "intent": "gratitude"},
    {"text": "I'm grateful for this space to talk",             "intent": "gratitude"},
    {"text": "I feel lighter after talking to you",             "intent": "gratitude"},
    {"text": "Thank you for not judging me",                    "intent": "gratitude"},
    {"text": "That really resonated with me thank you",         "intent": "gratitude"},
    {"text": "I feel less alone now",                           "intent": "gratitude"},
    {"text": "I appreciate that you remembered what I shared",  "intent": "gratitude"},
    {"text": "Your question made me think thank you",           "intent": "gratitude"},
    {"text": "I needed to hear that so much",                   "intent": "gratitude"},
    {"text": "This is exactly what I needed today",             "intent": "gratitude"},
    {"text": "Thank you for being patient with me",             "intent": "gratitude"},
    {"text": "I feel validated and that helps more than you know","intent": "gratitude"},
    {"text": "You always know what to say",                     "intent": "gratitude"},
    {"text": "I'm glad I came back to talk today",              "intent": "gratitude"},
    {"text": "Last session really changed my perspective",      "intent": "gratitude"},
    {"text": "I feel more hopeful after talking",               "intent": "gratitude"},
    {"text": "You help me understand myself better",            "intent": "gratitude"},
    {"text": "I'm grateful you don't give up on me",            "intent": "gratitude"},
    {"text": "Thank you for everything",                        "intent": "gratitude"},
    {"text": "I feel safe talking to you",                      "intent": "gratitude"},

    # ── progress ────────────────────────────────────────────────────────────
    {"text": "I actually had a good week for once",             "intent": "progress"},
    {"text": "I stood up for myself today and it felt amazing", "intent": "progress"},
    {"text": "I've been sleeping better lately",                "intent": "progress"},
    {"text": "I went outside today for the first time in a while","intent": "progress"},
    {"text": "I said no to someone and didn't feel guilty",     "intent": "progress"},
    {"text": "I finished something I'd been putting off",       "intent": "progress"},
    {"text": "I feel like I'm making real progress",            "intent": "progress"},
    {"text": "I talked to my mum and it actually went well",    "intent": "progress"},
    {"text": "I managed my anxiety without spiralling",         "intent": "progress"},
    {"text": "I've been journalling and it's helping",          "intent": "progress"},
    {"text": "I ate three proper meals today",                  "intent": "progress"},
    {"text": "I stopped a negative thought before it spiralled","intent": "progress"},
    {"text": "I reached out to a friend instead of isolating",  "intent": "progress"},
    {"text": "I completed my assignment and feel proud",        "intent": "progress"},
    {"text": "I had a moment of genuine happiness today",       "intent": "progress"},
    {"text": "I feel stronger than I did last month",           "intent": "progress"},
    {"text": "I finally started therapy",                       "intent": "progress"},
    {"text": "I exercised for the first time in months",        "intent": "progress"},
    {"text": "I talked about my feelings instead of bottling up","intent": "progress"},
    {"text": "I had a difficult conversation and handled it well","intent": "progress"},
    {"text": "I've been practising the things we talked about", "intent": "progress"},
    {"text": "I'm starting to recognise my own patterns",       "intent": "progress"},
    {"text": "I let myself feel proud today",                   "intent": "progress"},
    {"text": "I set a boundary with my family and held it",     "intent": "progress"},
    {"text": "Today I felt like myself for the first time in ages","intent": "progress"},

    # ── checking_in ─────────────────────────────────────────────────────────
    {"text": "Hey just checking in",                            "intent": "checking_in"},
    {"text": "Hi how are you",                                  "intent": "checking_in"},
    {"text": "Good morning",                                    "intent": "checking_in"},
    {"text": "I just wanted to say hello",                      "intent": "checking_in"},
    {"text": "Hey it's me again",                               "intent": "checking_in"},
    {"text": "Hello I'm back",                                  "intent": "checking_in"},
    {"text": "Just popping in",                                 "intent": "checking_in"},
    {"text": "Hi I haven't been on in a while",                 "intent": "checking_in"},
    {"text": "Good evening",                                    "intent": "checking_in"},
    {"text": "Hey I thought I'd check in today",               "intent": "checking_in"},
    {"text": "Hello again",                                     "intent": "checking_in"},
    {"text": "Just wanted to touch base",                       "intent": "checking_in"},
    {"text": "Hi there",                                        "intent": "checking_in"},
    {"text": "Hey",                                             "intent": "checking_in"},
    {"text": "Good to be back",                                 "intent": "checking_in"},
    {"text": "I'm back",                                        "intent": "checking_in"},
    {"text": "Hello just wanted to connect",                    "intent": "checking_in"},
    {"text": "Hey I've been meaning to come back",              "intent": "checking_in"},
    {"text": "Good afternoon",                                  "intent": "checking_in"},
    {"text": "Hi just wanted to check in with you",             "intent": "checking_in"},
    {"text": "Hey good to see you",                             "intent": "checking_in"},
    {"text": "Hello how have you been",                         "intent": "checking_in"},
    {"text": "Just saying hi",                                  "intent": "checking_in"},
    {"text": "Hey I'm here",                                    "intent": "checking_in"},
    {"text": "Hi just dropping by",                             "intent": "checking_in"},

    # ── general ─────────────────────────────────────────────────────────────
    {"text": "I had a strange day",                             "intent": "general"},
    {"text": "Not sure where to start",                         "intent": "general"},
    {"text": "I've been thinking a lot lately",                 "intent": "general"},
    {"text": "Something feels off but I can't place it",        "intent": "general"},
    {"text": "I don't really know how I'm feeling",             "intent": "general"},
    {"text": "I just needed to talk to someone",                "intent": "general"},
    {"text": "I'm not sure what's going on with me",            "intent": "general"},
    {"text": "I feel strange today and I can't explain it",     "intent": "general"},
    {"text": "I've been a bit all over the place recently",     "intent": "general"},
    {"text": "I wanted to talk but I'm not sure about what",    "intent": "general"},
    {"text": "Something happened today that I can't explain",   "intent": "general"},
    {"text": "I don't know where my head's at",                 "intent": "general"},
    {"text": "I just felt like reaching out",                   "intent": "general"},
    {"text": "I haven't been myself lately",                    "intent": "general"},
    {"text": "Things have been complicated",                    "intent": "general"},
    {"text": "I've been a bit lost recently",                   "intent": "general"},
    {"text": "I'm not really sure how to describe what I feel", "intent": "general"},
    {"text": "I just want to process out loud",                 "intent": "general"},
    {"text": "I feel like something has shifted but I don't know what","intent": "general"},
    {"text": "I needed some space to think and talk",           "intent": "general"},
    {"text": "I'm kind of just drifting right now",             "intent": "general"},
    {"text": "I feel somewhere between okay and not okay",      "intent": "general"},
    {"text": "I'm in a weird headspace lately",                 "intent": "general"},
    {"text": "Things feel uncertain and I don't like it",       "intent": "general"},
    {"text": "I just want to talk through some things",         "intent": "general"},
]


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"([^\w\s'])", r" \1 ", text)
    return text.split()


_PERSON  = {"mum","mom","dad","father","mother","brother","sister","friend","partner",
            "boyfriend","girlfriend","husband","wife","boss","colleague","teacher","flatmate",
            "sister","parents","family","manager"}
_TOPIC   = {"work","job","career","office","exam","deadline","school","university","uni",
            "sleep","health","relationship","money","studies","dissertation","presentation",
            "commute","therapy","journalling"}
_ISSUE   = {"anxiety","depression","trauma","abuse","panic","hopeless","suicide","cutting",
            "self-harm","ptsd","burnout","ptsd","flashbacks","nightmares","dissociate"}
_MOOD    = {"anxious","sad","angry","happy","hopeless","numb","empty","overwhelmed",
            "exhausted","furious","pissed","grateful","better","scared","terrified",
            "devastated","frustrated","lonely","worthless","guilty","ashamed","proud"}


def _tag(tokens: list[str]) -> list[str]:
    tags, prev = [], "O"
    for tok in tokens:
        t = tok.strip("'.,!?")
        if t in _PERSON:
            tag = "B-PERSON" if prev != "I-PERSON" else "I-PERSON"; prev = "I-PERSON"
        elif t in _TOPIC:
            tag = "B-TOPIC"  if prev != "I-TOPIC"  else "I-TOPIC";  prev = "I-TOPIC"
        elif t in _ISSUE:
            tag = "B-ISSUE"  if prev != "I-ISSUE"  else "I-ISSUE";  prev = "I-ISSUE"
        elif t in _MOOD:
            tag = "B-MOOD"   if prev != "I-MOOD"   else "I-MOOD";   prev = "I-MOOD"
        else:
            tag = "O"; prev = "O"
        tags.append(tag)
    return tags


def main():
    random.seed(42)
    out = Path("training/data")
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for item in SEED_DATA:
        tokens = _tokenize(item["text"])
        records.append({
            "text":     item["text"],
            "intent":   item["intent"],
            "tokens":   tokens,
            "dst_tags": _tag(tokens),
        })

    random.shuffle(records)
    split = int(len(records) * 0.85)
    write = lambda p, d: Path(p).write_text("\n".join(json.dumps(r) for r in d))
    write(out / "multitask_train.jsonl", records[:split])
    write(out / "multitask_val.jsonl",   records[split:])

    from collections import Counter
    intents = Counter(r["intent"] for r in records)
    print(f"Total: {len(records)}  Train: {split}  Val: {len(records)-split}")
    print("Per class:", dict(sorted(intents.items())))


if __name__ == "__main__":
    main()
