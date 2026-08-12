import sys
sys.dont_write_bytecode = True

from typing import List
from pydantic import BaseModel, Field

RAG_K_THRESHOLD = 5


class ApplicantID(BaseModel):
    id_list: List[str] = Field(
        ...,
        description="List of IDs of the applicants to retrieve resumes for"
    )


class JobDescription(BaseModel):
    job_description: str = Field(
        ...,
        description="Description of a job to retrieve similar resumes for"
    )


class RAGRetriever():
    def __init__(self, vectorstore_db, df):
        self.vectorstore = vectorstore_db
        self.df = df

    def reciprocal_rank_fusion(self, document_rank_list: list[dict], k=50):
        fused_scores = {}

        for doc_list in document_rank_list:
            for rank, (doc, _) in enumerate(doc_list.items()):
                if doc not in fused_scores:
                    fused_scores[doc] = 0

                fused_scores[doc] += 1 / (rank + k)

        reranked_results = {
            doc: score
            for doc, score in sorted(
                fused_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
        }

        return reranked_results

    def retrieve_docs_id(self, question: str, k=50):
        docs_score = self.vectorstore.similarity_search_with_score(
            question,
            k=k
        )

        docs_score = {
            str(doc.metadata["ID"]): score
            for doc, score in docs_score
        }

        return docs_score

    def retrieve_id_and_rerank(self, subquestion_list: list):
        document_rank_list = []

        for subquestion in subquestion_list:
            document_rank_list.append(
                self.retrieve_docs_id(
                    subquestion,
                    RAG_K_THRESHOLD
                )
            )

        reranked_documents = self.reciprocal_rank_fusion(
            document_rank_list
        )

        return reranked_documents

    def retrieve_documents_with_id(
        self,
        doc_id_with_score: dict,
        threshold=5
    ):
        id_resume_dict = dict(
            zip(
                self.df["ID"].astype(str),
                self.df["Resume"]
            )
        )

        retrieved_ids = list(
            sorted(
                doc_id_with_score,
                key=doc_id_with_score.get,
                reverse=True
            )
        )[:threshold]

        retrieved_documents = [
            id_resume_dict[id]
            for id in retrieved_ids
        ]

        for i in range(len(retrieved_documents)):
            retrieved_documents[i] = (
                "Applicant ID "
                + retrieved_ids[i]
                + "\n"
                + retrieved_documents[i]
            )

        return retrieved_documents


class SelfQueryRetriever(RAGRetriever):
    def __init__(self, vectorstore_db, df):
        super().__init__(vectorstore_db, df)

        self.meta_data = {
            "rag_mode": "",
            "query_type": "no_retrieve",
            "extracted_input": "",
            "subquestion_list": [],
            "retrieved_docs_with_scores": []
        }

    def retrieve_applicant_id(self, id_list: list):
        retrieved_resumes = []

        for applicant_id in id_list:
            try:
                resume_df = self.df[
                    self.df["ID"].astype(str) == str(applicant_id)
                ].iloc[0][["ID", "Resume"]]

                resume_with_id = (
                    "Applicant ID "
                    + resume_df["ID"].astype(str)
                    + "\n"
                    + resume_df["Resume"]
                )

                retrieved_resumes.append(resume_with_id)

            except Exception:
                return []

        return retrieved_resumes

    def _find_applicant_ids(self, question: str):
        question_lower = question.lower()
        applicant_ids = []

        for applicant_id in self.df["ID"].astype(str).unique():
            if str(applicant_id).lower() in question_lower:
                applicant_ids.append(str(applicant_id))

        return applicant_ids

    def retrieve_applicant_jd(
        self,
        question: str,
        llm,
        rag_mode: str
    ):
        subquestion_list = [question]

        if rag_mode == "RAG Fusion":
            subquestion_list += llm.generate_subquestions(question)

        self.meta_data["subquestion_list"] = subquestion_list

        retrieved_ids = self.retrieve_id_and_rerank(
            subquestion_list
        )

        self.meta_data["retrieved_docs_with_scores"] = retrieved_ids

        retrieved_resumes = self.retrieve_documents_with_id(
            retrieved_ids
        )

        return retrieved_resumes

    def retrieve_docs(
        self,
        question: str,
        llm,
        rag_mode: str
    ):
        self.meta_data["rag_mode"] = rag_mode
        self.meta_data["query_type"] = "no_retrieve"
        self.meta_data["extracted_input"] = question

        applicant_ids = self._find_applicant_ids(question)

        if applicant_ids:
            self.meta_data["query_type"] = "retrieve_applicant_id"
            self.meta_data["extracted_input"] = {
                "id_list": applicant_ids
            }

            return self.retrieve_applicant_id(
                applicant_ids
            )

        self.meta_data["query_type"] = "retrieve_applicant_jd"
        self.meta_data["extracted_input"] = {
            "job_description": question
        }

        return self.retrieve_applicant_jd(
            question,
            llm,
            rag_mode
        )