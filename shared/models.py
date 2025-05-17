from pydantic import BaseModel, Field, TypeAdapter


class EvalResponse(BaseModel):
    question: str = Field(default=None, init=False)
    ai_answer: str
    ideal_answer: str
    evaluation: str
    context: str = Field(default=None, init=False)
    hash: str = Field(default=None, init=False)
    score: int = Field(ge=0, le=1)


class QAItem(BaseModel):
    question: str
    ideal_answer: str


qa_list_adapter = TypeAdapter(list[QAItem])
