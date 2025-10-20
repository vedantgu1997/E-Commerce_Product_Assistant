from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
import asyncio


from product_assistant.prompt_library.prompts import PROMPT_REGISTRY, PromptType    
from product_assistant.retriever.retrieval import Retriever
from product_assistant.utils.model_loader import ModelLoader
from product_assistant.evaluation.ragas_eval import evaluate_response_precision, evaluate_response_relevancy


class AgenticRAG:
    """Agentic RAG pipeline using LangGraph."""

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def __init__(self):
        self.retriver_obj = Retriever()
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.checkpointer = MemorySaver()
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    # ---------------- Helpers ----------------
    def _format_docs(self, docs) -> str:
        if not docs:
            return "No relevant documents found."
        formatted_chunks = []
        for d in docs:
            meta = d.metadata or {}
            formatted = (
                f"Title: {meta.get('product_title', 'N/A')}\n"
                f"Price: {meta.get('price', 'N/A')}\n"
                f"Rating: {meta.get('rating', 'N/A')}\n"
                f"Reviews: \n{d.page_content.strip()}"
            )
            formatted_chunks.append(formatted)
        return "\n\n---\n\n".join(formatted_chunks)
    
    # ---------------- Nodes ----------------
    def _ai_assistant(self, state: AgentState):
        """Decide whether to call the retriever or answer directly."""
        print("--- Calling AI Assistant Node ---")
        messages = state["messages"]
        last_message = messages[-1].content

        # Simple routing: if query mentions product -> go retriever
        if any(word in last_message.lower() for word in ["product", "price", "review"]): #type: ignore
            return {"messages": [HumanMessage(content="TOOL: retriever")]}
        else:
            prompt = ChatPromptTemplate.from_template(
                "You are a helpful assistant. Answer the user directly.\n\nQuestion: {question}\nAnswer:"
            )
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": last_message})
            return {"messages": [HumanMessage(content=response)]}
        
    def _vector_retriever(self, state: AgentState):
        """Fetch product info from vector DB."""
        print("--- RETRIEVER ---")
        query = state["messages"][-1].content
        retriever = self.retriver_obj.load_retriever()
        docs = retriever.invoke(query)  # type: ignore
        context = self._format_docs(docs)
        response_message = HumanMessage(content=f"CONTEXT: {context}\n\nQuestion: {query}\nAnswer:")
        return {"messages": [response_message]}
    
    def _grade_documents(self, state: AgentState) -> Literal["generator", "rewriter"]:
        """Grade docs relevance"""
        print("--- GRADER ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content
        prompt = ChatPromptTemplate.from_template(
            "Given the question and the retrieved documents, decide if the documents are relevant enough to answer the question. "
            "If they are relevant, return 'generator'. If they are not relevant, return 'rewriter'.\n\n"
            "Question: {question}\nDocuments: {documents}\nDecision:"
        )
        chain = prompt | self.llm | StrOutputParser()
        score = chain.invoke({"question": question, "documents": docs})
        return "generator" if "yes" in score.lower() else "rewriter"
    
    def _generate(self, state: AgentState):
        """Generate answer using LLM and retrieved context."""
        print("--- GENERATOR ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content
        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
        )
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"question": question, "context": docs})
        return {"messages": [HumanMessage(content=answer)]}
    
    def _rewrite(self, state: AgentState):
        """Rewrite question for clarity."""
        """Rewrite bad query"""
        print("--- REWRITE ---")
        question = state["messages"][0].content
        new_q = self.llm.invoke(
            [HumanMessage(content=f"Rewrite this question to be more specific: {question}")]
        )
        return {"messages": [HumanMessage(content=new_q.content)]}

    # ---------------- Build Workflow ----------------
    def _build_workflow(self):
        workflow = StateGraph(self.AgentState)
        workflow.add_node("Assistant", self._ai_assistant)
        workflow.add_node("Retriever", self._vector_retriever)
        workflow.add_node("Grader", self._grade_documents)
        workflow.add_node("Generator", self._generate) 
        workflow.add_node("Rewriter", self._rewrite)

        workflow.add_edge(START, "Assistant")
        workflow.add_conditional_edges(
            "Assistant",
            lambda state: "Retriever" if "TOOL" in state["messages"][-1].content else END,
            {"Retriever": "Retriever", END: END}
        )
        workflow.add_conditional_edges(
            "Retriever",
            self._grade_documents,
            {"Generator": "Generator", "Rewriter": "Rewriter"}
        )
        workflow.add_edge("Generator", END)
        workflow.add_edge("Rewriter", "Assistant")
        return workflow
    
    # ---------------- Public Run ----------------
    def run(self, query: str, thread_id: str = "default_thread"):
        """Run the agentic RAG workflow."""
        result = self.app.invoke({"messages": [HumanMessage(content=query)]},
                                 config={"configurable":{'thread_id': thread_id}})
        return result["messages"][-1].content
    
if __name__ == "__main__":
    rag_agent = AgenticRAG()
    answer = rag_agent.run("Can you tell me the price of iphone 15?")
    print("\n Assistant Answer:\n", answer)