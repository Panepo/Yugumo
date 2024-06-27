from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForToolRun
from typing import Optional


class WikipediaQueryRunWrapper(WikipediaQueryRun):
    def _run(
        self,
        text: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Wikipedia tool."""
        return self.api_wrapper.run(text)


api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)


class WikiInputs(BaseModel):
    """inputs to the wikipedia tool."""

    text: str = Field(description="query to look up on wikipedia.")


wikipedia = WikipediaQueryRunWrapper(
    description="A wrapper around Wikipedia. Useful for when you need to answer general questions about people, places, companies, facts, historical events, or other subjects. Input should be a search query.",
    args_schema=WikiInputs,
    api_wrapper=api_wrapper,
)
