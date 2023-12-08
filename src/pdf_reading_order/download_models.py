from huggingface_hub import hf_hub_download

candidate_selector_model = hf_hub_download(
    repo_id="HURIDOCS/pdf-reading-order",
    filename="candidate_selector_model.model",
    revision="4117935c3500d58eca15ca89dfba211e5c73ae45",
)

reading_order_model = hf_hub_download(
    repo_id="HURIDOCS/pdf-reading-order",
    filename="reading_order_model.model",
    revision="17cf6f396cfd39d2290d70264f97b640c9f5b5c7",
)
