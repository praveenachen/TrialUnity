import os
import logging

from openai import AsyncOpenAI

from backend.app.core.config import get_settings
from backend.app.models.schemas import AssistantRequest, AssistantResponse
from backend.app.services.text import first_sentence

logger = logging.getLogger("trialunity.assistant")


class TrialAssistant:
    async def answer(self, request: AssistantRequest) -> AssistantResponse:
        settings = get_settings()
        api_key = os.getenv("OPENAI_API_KEY") or settings.openai_api_key
        sources = [request.trial.nct_id]
        if request.trial.source_url:
            sources.append(request.trial.source_url)

        if not settings.enable_llm or not api_key:
            logger.info("assistant_provider=fallback reason=missing_llm_config")
            return AssistantResponse(answer=self._fallback_answer(request), provider="fallback", sources=sources)

        client = AsyncOpenAI(api_key=api_key)
        prompt = self._grounded_prompt(request)
        response = await client.responses.create(
            model=settings.openai_model,
            input=prompt,
            temperature=0.2,
        )
        logger.info("assistant_provider=openai model=%s trial=%s", settings.openai_model, request.trial.nct_id)
        return AssistantResponse(answer=response.output_text, provider="openai", sources=sources)

    def _fallback_answer(self, request: AssistantRequest) -> str:
        trial = request.trial
        summary = first_sentence(trial.brief_summary)
        eligibility = first_sentence(trial.eligibility_criteria, "The study eligibility text is not available.")
        return (
            "### Fit summary\n"
            f"- {summary}\n\n"
            "### Eligibility checklist\n"
            f"- {eligibility}\n\n"
            "### What to confirm\n"
            "- Confirm full eligibility with the study team before outreach.\n"
            "- This explanation is not medical advice."
        )

    def _grounded_prompt(self, request: AssistantRequest) -> str:
        trial = request.trial
        profile = request.patient_profile.model_dump() if request.patient_profile else {}
        return f"""
You are a careful clinical trial navigation assistant. Answer only from the trial data below.
If the data is missing, say what is missing. Use patient-friendly language and avoid medical advice.

Return a concise navigation note using exactly these Markdown headings:

### Fit summary
- 1-2 bullets explaining why this trial may be relevant to the patient profile.

### Eligibility checklist
- 2-4 bullets using only eligibility details present in the trial record.
- Clearly say "Not specified in the trial record" when age, biomarkers, prior treatments, location, or other details are missing.

### What to confirm
- 2-3 bullets listing the most important questions for the patient or clinician to ask the study team.

### Patient-friendly note
- One short sentence reminding the user this is trial navigation support, not medical advice.

Avoid numbering, long paragraphs, and unsupported claims.

Question: {request.question}
Patient profile: {profile}

Trial:
NCT ID: {trial.nct_id}
Title: {trial.title}
Status: {trial.status}
Conditions: {', '.join(trial.conditions)}
Interventions: {', '.join(trial.interventions)}
Phases: {', '.join(trial.phases)}
Summary: {trial.brief_summary}
Eligibility: {trial.eligibility_criteria}
Locations: {', '.join(trial.locations)}
""".strip()
