"""The hiring-platform demo catalogue: domain, MCP tools, intents, examples.

Kept as data so it can be reviewed and re-run. Intent names are deliberately
close to one another in places -- CANDIDATE_SEARCH beside JOB_SEARCH,
INTERVIEW_SCHEDULE beside INTERVIEW_RESCHEDULE -- because a real applicant
tracking system looks like that, and separating them is the part worth showing.
"""

from __future__ import annotations

DOMAIN = {
    "name": "hiring",
    "description": (
        "an applicant tracking system: open requisitions, search and shortlist "
        "candidates, schedule interviews, collect scorecards, make offers and "
        "report on the pipeline"
    ),
}

#: An ATS server's tools/list response, as a real MCP server would report it.
MCP_SERVER = {
    "server": "ats",
    "transport": "http",
    "endpoint": "https://ats.internal/mcp",
    "tools": {
        "tools": [
            {
                "name": "search_candidates",
                "description": "Find candidates by skill, location, seniority or stage",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "skill": {"type": "string"},
                        "location": {"type": "string"},
                        "seniority": {"type": "string"},
                        "stage": {"type": "string"},
                    },
                    "required": [],
                },
            },
            {
                "name": "get_candidate",
                "description": "Fetch one candidate's full profile",
                "inputSchema": {
                    "type": "object",
                    "properties": {"candidate_id": {"type": "string"}},
                    "required": ["candidate_id"],
                },
            },
            {
                "name": "compare_candidates",
                "description": "Put two candidates side by side",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "other_candidate_id": {"type": "string"},
                    },
                    "required": ["candidate_id"],
                },
            },
            {
                "name": "shortlist_candidate",
                "description": "Add a candidate to a requisition's shortlist",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "job_id": {"type": "string"},
                    },
                    "required": ["candidate_id"],
                },
            },
            {
                "name": "reject_candidate",
                "description": "Reject a candidate with a reason",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["candidate_id"],
                },
            },
            {
                "name": "create_requisition",
                "description": "Open a new requisition",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "location": {"type": "string"},
                        "seniority": {"type": "string"},
                        "headcount": {"type": "integer"},
                    },
                    "required": ["title"],
                },
            },
            {
                "name": "search_requisitions",
                "description": "Find requisitions by status, team or location",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "team": {"type": "string"},
                        "location": {"type": "string"},
                    },
                    "required": [],
                },
            },
            {
                "name": "get_requisition_status",
                "description": "Pipeline status for one requisition",
                "inputSchema": {
                    "type": "object",
                    "properties": {"job_id": {"type": "string"}},
                    "required": ["job_id"],
                },
            },
            {
                "name": "close_requisition",
                "description": "Close a requisition",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["job_id"],
                },
            },
            {
                "name": "schedule_interview",
                "description": "Book an interview",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "interviewer": {"type": "string"},
                        "round": {"type": "string"},
                        "interview_date": {"type": "string"},
                    },
                    "required": ["candidate_id"],
                },
            },
            {
                "name": "reschedule_interview",
                "description": "Move an existing interview",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "interview_date": {"type": "string"},
                    },
                    "required": ["candidate_id"],
                },
            },
            {
                "name": "get_interview_feedback",
                "description": "Read interview scorecards",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "round": {"type": "string"},
                    },
                    "required": ["candidate_id"],
                },
            },
            {
                "name": "create_offer",
                "description": "Raise an offer for a candidate",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "job_id": {"type": "string"},
                        "salary": {"type": "number"},
                    },
                    "required": ["candidate_id"],
                },
            },
            {
                "name": "get_offer_status",
                "description": "Where an offer has reached",
                "inputSchema": {
                    "type": "object",
                    "properties": {"candidate_id": {"type": "string"}},
                    "required": ["candidate_id"],
                },
            },
            {
                "name": "advance_application",
                "description": "Move an application to the next stage",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "stage": {"type": "string"},
                    },
                    "required": ["candidate_id"],
                },
            },
            {
                "name": "get_pipeline_report",
                "description": "Funnel counts and conversion for a requisition or team",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string"},
                        "team": {"type": "string"},
                        "period": {"type": "string"},
                    },
                    "required": [],
                },
            },
            {
                "name": "submit_referral",
                "description": "Refer someone for a role",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "candidate_name": {"type": "string"},
                        "job_id": {"type": "string"},
                    },
                    "required": ["candidate_name"],
                },
            },
            {
                "name": "get_sourcing_analytics",
                "description": "Where candidates are coming from",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "period": {"type": "string"},
                    },
                    "required": [],
                },
            },
        ]
    },
}

STAGE = {
    "type": "enum",
    "values": ["APPLIED", "SCREENING", "INTERVIEW", "OFFER", "HIRED", "REJECTED"],
}
SENIORITY = {"type": "enum", "values": ["INTERN", "JUNIOR", "SENIOR", "STAFF", "PRINCIPAL"]}
REQ_STATUS = {"type": "enum", "values": ["OPEN", "ON_HOLD", "CLOSED", "FILLED"]}

#: name, description, mcp tool, entity schema, extraction hints, examples.
INTENTS: list[dict] = [
    {
        "name": "CANDIDATE_SEARCH",
        "description": "Find candidates matching skills, location, seniority or pipeline stage.",
        "tool": "search_candidates",
        "entity_schema": {
            "skill": {"type": "string"},
            "location": {"type": "string"},
            "seniority": SENIORITY,
            "stage": STAGE,
        },
        "extraction_hints": "'Bangalore', 'Berlin' and similar are locations, not names.",
        "examples": [
            "find senior backend engineers in Bangalore",
            "show me candidates who know Kubernetes",
            "search for staff engineers in Berlin",
            "which candidates are at the screening stage",
            "list frontend candidates in Dublin",
            "any principal engineers in the pipeline",
            "pull up candidates with Golang experience",
            "look for junior data scientists in Pune",
            "who have we got for machine learning roles",
        ],
    },
    {
        "name": "CANDIDATE_PROFILE",
        "description": "Open one candidate's full profile.",
        "tool": "get_candidate",
        "entity_schema": {"candidate_id": {"type": "string", "required": True}},
        "extraction_hints": "Candidate ids look like C-2201.",
        "examples": [
            "open candidate C-2201",
            "show me the profile for C-3310",
            "pull up candidate C-2201's details",
            "I want to see C-4102's resume",
            "view the full profile of candidate C-5567",
            "what does candidate C-2201 look like",
            "display candidate C-3310",
            "get me the record for C-7781",
        ],
    },
    {
        "name": "CANDIDATE_COMPARE",
        "description": "Put two candidates side by side.",
        "tool": "compare_candidates",
        "entity_schema": {
            "candidate_id": {"type": "string", "required": True},
            "other_candidate_id": {"type": "string"},
        },
        "examples": [
            "compare candidate C-2201 with C-3310",
            "put C-4102 and C-5567 side by side",
            "how do C-2201 and C-3310 differ",
            "which is stronger, C-4102 or C-7781",
            "diff these two candidates",
            "show me C-2201 against C-5567",
            "contrast candidate C-3310 with C-4102",
            "weigh up C-7781 versus C-2201",
        ],
    },
    {
        "name": "CANDIDATE_SHORTLIST",
        "description": "Add a candidate to a requisition's shortlist.",
        "tool": "shortlist_candidate",
        "entity_schema": {
            "candidate_id": {"type": "string", "required": True},
            "job_id": {"type": "string"},
        },
        "examples": [
            "shortlist candidate C-2201 for J-4471",
            "add C-3310 to the shortlist",
            "put candidate C-4102 forward for the staff role",
            "move C-5567 onto the shortlist for J-4471",
            "I want to shortlist C-7781",
            "mark candidate C-2201 as shortlisted",
            "add this candidate to J-4471's shortlist",
            "please shortlist C-3310 for the backend req",
        ],
    },
    {
        "name": "CANDIDATE_REJECT",
        "description": "Reject a candidate, with a reason.",
        "tool": "reject_candidate",
        "entity_schema": {
            "candidate_id": {"type": "string", "required": True},
            "reason": {"type": "string"},
        },
        "examples": [
            "reject candidate C-2201",
            "turn down C-3310, not enough depth",
            "we are passing on candidate C-4102",
            "decline C-5567 for this role",
            "mark C-7781 as rejected",
            "send a rejection to candidate C-2201",
            "no further action on C-3310",
            "close out candidate C-4102 as unsuccessful",
        ],
    },
    {
        "name": "JOB_CREATE",
        "description": "Open a new requisition.",
        "tool": "create_requisition",
        "entity_schema": {
            "title": {"type": "string"},
            "location": {"type": "string"},
            "seniority": SENIORITY,
            "headcount": {"type": "integer"},
        },
        "examples": [
            "open a new requisition for a senior backend engineer",
            "create a job opening in Bangalore",
            "raise a req for two platform engineers",
            "I need to open a staff engineer role in Berlin",
            "start a new requisition for a data scientist",
            "set up a job posting for a junior designer",
            "create a requisition for 3 support engineers",
            "open a role for a principal architect in Dublin",
        ],
    },
    {
        "name": "JOB_SEARCH",
        "description": "Find requisitions by status, team or location.",
        "tool": "search_requisitions",
        "entity_schema": {
            "status": REQ_STATUS,
            "team": {"type": "string"},
            "location": {"type": "string"},
        },
        "extraction_hints": "A team is an organisational unit such as Platform or Payments.",
        "examples": [
            "which requisitions are open right now",
            "list the roles we are hiring for in Bangalore",
            "show me all open reqs on the Platform team",
            "what jobs are on hold",
            "find requisitions for the Payments team",
            "show me closed requisitions",
            "which openings does Infrastructure have",
            "list every filled requisition this quarter",
        ],
    },
    {
        "name": "JOB_STATUS",
        "description": "Pipeline status for one requisition.",
        "tool": "get_requisition_status",
        "entity_schema": {"job_id": {"type": "string", "required": True}},
        "extraction_hints": "Requisition ids look like J-4471.",
        "examples": [
            "what is the status of requisition J-4471",
            "how is J-6620 doing",
            "where has req J-4471 got to",
            "give me the state of J-8890",
            "is J-6620 still open",
            "status check on requisition J-4471",
            "how many people are in the pipeline for J-8890",
            "tell me where J-6620 stands",
        ],
    },
    {
        "name": "JOB_CLOSE",
        "description": "Close a requisition.",
        "tool": "close_requisition",
        "entity_schema": {
            "job_id": {"type": "string", "required": True},
            "reason": {"type": "string"},
        },
        "examples": [
            "close requisition J-4471",
            "we can shut down J-6620",
            "cancel the req J-8890",
            "mark J-4471 as closed, budget pulled",
            "stop hiring for J-6620",
            "close out requisition J-8890",
            "put J-4471 to bed, role is filled",
            "withdraw requisition J-6620",
        ],
    },
    {
        "name": "INTERVIEW_SCHEDULE",
        "description": "Book an interview for a candidate.",
        "tool": "schedule_interview",
        "entity_schema": {
            "candidate_id": {"type": "string", "required": True},
            "interviewer": {"type": "string"},
            "round": {"type": "string"},
            "interview_date": {"type": "date"},
        },
        "examples": [
            "schedule a new interview for candidate C-2201",
            "book a first system design round for C-3310",
            "set up an interview with C-4102 on March 3, 2026",
            "arrange a screening call for candidate C-5567",
            "put C-7781 in for a technical round",
            "book the final round for C-2201",
            "schedule C-3310 with the hiring manager next Tuesday",
            "get a new interview in the diary for C-4102",
            "line up an interview for candidate C-7781",
            "set a date for C-2201's design round",
        ],
    },
    {
        "name": "INTERVIEW_RESCHEDULE",
        "description": "Move an interview that is already booked.",
        "tool": "reschedule_interview",
        "entity_schema": {
            "candidate_id": {"type": "string", "required": True},
            "interview_date": {"type": "date"},
        },
        "examples": [
            "reschedule the interview we already booked for C-2201",
            "move C-3310's existing interview to March 10, 2026",
            "push back the round already in the diary for C-4102",
            "shift C-5567's booked slot to next week",
            "we need to rearrange the interview C-7781 already has",
            "the time no longer works, move C-2201's interview",
            "postpone the interview that is booked with C-3310",
            "bring C-4102's already-scheduled interview forward",
            "C-5567 cannot make the booked slot, find another",
        ],
    },
    {
        "name": "INTERVIEW_FEEDBACK",
        "description": "Read the scorecards from a candidate's interviews.",
        "tool": "get_interview_feedback",
        "entity_schema": {
            "candidate_id": {"type": "string", "required": True},
            "round": {"type": "string"},
        },
        "examples": [
            "what did the interviewers say about C-2201",
            "show me the scorecards for candidate C-3310",
            "pull the feedback for C-4102's design round",
            "how did C-5567 do in the interview",
            "read me the interview notes on C-7781",
            "what was the verdict on candidate C-2201",
            "get the panel feedback for C-3310",
            "were the reviews positive for C-4102",
        ],
    },
    {
        "name": "OFFER_CREATE",
        "description": "Raise an offer for a candidate.",
        "tool": "create_offer",
        "entity_schema": {
            "candidate_id": {"type": "string", "required": True},
            "job_id": {"type": "string"},
            "salary": {"type": "number"},
        },
        "examples": [
            "raise an offer for candidate C-2201",
            "make an offer to C-3310 for J-4471",
            "let us offer C-4102 the staff role",
            "prepare an offer letter for candidate C-5567",
            "extend an offer to C-7781",
            "put together an offer for C-2201 at 4200000",
            "we want to offer candidate C-3310",
            "start the offer process for C-4102",
        ],
    },
    {
        "name": "OFFER_STATUS",
        "description": "Where a candidate's offer has reached.",
        "tool": "get_offer_status",
        "entity_schema": {"candidate_id": {"type": "string", "required": True}},
        "examples": [
            "has candidate C-2201 accepted yet",
            "what is the offer status for C-3310",
            "did C-4102 sign",
            "where is the offer for candidate C-5567",
            "is C-7781's offer still pending approval",
            "check the offer state for C-2201",
            "have we heard back from C-3310 on the offer",
            "track the offer for candidate C-4102",
        ],
    },
    {
        "name": "APPLICATION_ADVANCE",
        "description": "Move an application to the next stage of the pipeline.",
        "tool": "advance_application",
        "entity_schema": {
            "candidate_id": {"type": "string", "required": True},
            "stage": STAGE,
        },
        "examples": [
            "move candidate C-2201 to the interview stage",
            "advance C-3310 to OFFER",
            "progress candidate C-4102 to screening",
            "push C-5567 through to the next round",
            "take C-7781 forward to interview",
            "promote candidate C-2201 to the offer stage",
            "move C-3310 along the pipeline",
            "mark C-4102 as hired",
        ],
    },
    {
        "name": "PIPELINE_REPORT",
        "description": "Funnel counts and conversion for a requisition, team or period.",
        "tool": "get_pipeline_report",
        "entity_schema": {
            "job_id": {"type": "string"},
            "team": {"type": "string"},
            "period": {"type": "string"},
        },
        "examples": [
            "how is the hiring funnel looking this quarter",
            "give me a pipeline report for J-4471",
            "what is our conversion from screening to offer",
            "show funnel numbers for the Platform team",
            "how many candidates dropped off at interview",
            "pipeline breakdown for the Payments team",
            "report on our hiring throughput this month",
            "where are we losing candidates",
        ],
    },
    {
        "name": "REFERRAL_SUBMIT",
        "description": "Refer someone for an open role.",
        "tool": "submit_referral",
        "entity_schema": {
            "candidate_name": {"type": "string", "required": True},
            "job_id": {"type": "string"},
        },
        "extraction_hints": "A referral names a person, not a candidate id.",
        "examples": [
            "I want to refer Priya Raman for J-4471",
            "submit a referral for Daniel Okafor",
            "refer my former colleague Mei Lin",
            "put forward Anita Desai as a referral",
            "I am referring Tomas Novak for the platform role",
            "add a referral for Sarah Whitfield",
            "recommend Rahul Menon for J-6620",
            "refer Kofi Mensah for an engineering opening",
            "a friend of mine would be good for this, her name is Leila Haddad",
            "referring someone I used to work with, Jonas Berg",
            "I would like to refer a former teammate for an open role",
            "please log a referral, the person is Ana Ruiz",
        ],
    },
    {
        "name": "SOURCING_ANALYTICS",
        "description": "Where candidates are coming from, and which sources convert.",
        "tool": "get_sourcing_analytics",
        "entity_schema": {
            "source": {"type": "string"},
            "period": {"type": "string"},
        },
        "examples": [
            "which sourcing channel gives us the best candidates",
            "how many hires came from referrals this year",
            "break down applications by source",
            "is LinkedIn worth the spend",
            "compare agency hires against inbound",
            "what share of candidates come from job boards",
            "sourcing effectiveness for this quarter",
            "which channels convert to offers",
        ],
    },
]

#: The conversation the demo records, with what each turn is meant to show.
CONVERSATION = [
    (
        "find senior backend engineers in Bangalore",
        "A plain search. Skill, location and seniority are extracted and mapped onto the ATS tool.",
    ),
    (
        "open candidate C-2201",
        "A different intent entirely, distinguished from the search by wording, not by keywords.",
    ),
    (
        "compare them with C-3310",
        "A follow-up: 'them' points back, so it is read against the previous turn.",
    ),
    (
        "book C-2201 in for a technical round on March 3, 2026",
        "An action rather than a lookup, and close in wording to rescheduling. The date is "
        "normalised to ISO 8601.",
    ),
    (
        "what did the interviewers say",
        "Referential again. The candidate carries forward from earlier in the conversation.",
    ),
    (
        "raise an offer for C-2201",
        "Late-pipeline intent, close in wording to several others and still separated.",
    ),
    (
        "how is the hiring funnel looking this quarter",
        "Analytics, with no candidate involved. A different shape of question in the same domain.",
    ),
    (
        "I'd like to put forward a former colleague, Priya Raman, for the platform role",
        "A person's name rather than an id, phrased unlike any stored example. This is "
        "retrieval generalising rather than matching a string.",
    ),
    (
        "what is the weather today",
        "Out of scope. It must come back UNKNOWN rather than being absorbed by the conversation.",
    ),
]
