import asyncio
from utils.model_loader import ModelLoader
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference, ResponseRelevancy
# from grpc.experimental.aio import grpc_aio
# grpc_aio.init_grpc_aio()
model_loader = ModelLoader()

def evaluate_response_precision(query, response, retrieved_context):
    try:
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            reference_contexts=retrieved_context
        )

        async def main():
            llm = model_loader.load_llm()
            evaluator_llm = LangchainLLMWrapper(llm)
            context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)
            result = await context_precision.single_turn_ascore(sample)
            return result
        
        return asyncio.run(main())
    except Exception as e:
        return e
    

def evaluate_response_relevancy(query, response, retrieved_context):
    try:
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            reference_contexts=retrieved_context
        )

        async def main():
            llm = model_loader.load_llm()
            evaluator_llm = LangchainLLMWrapper(llm)
            relevancy_metric = ResponseRelevancy(llm=evaluator_llm)
            result = await relevancy_metric.single_turn_ascore(sample)
            return result
        
        return asyncio.run(main())
    except Exception as e:
        return e
