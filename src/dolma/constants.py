"""Dolma dataset identifiers and shared constants."""

DOLMA_DATASET_ID = "allenai/dolma3_mix-150B-1025"
DOLMA_6T_MIX_DATASET_ID = "allenai/dolma3_mix-6T-1025-7B"
DOLMA_DOLMINO_MIX_1025_DATASET_ID = "allenai/dolma3_dolmino_mix-100B-1025"
DOLMA_POOL_DATASET_ID = "allenai/dolma3_pool"
ALLOWED_DOLMA_DATASET_IDS = (DOLMA_DATASET_ID, DOLMA_POOL_DATASET_ID)
ALLOWED_DOLMA_SUBSETS: tuple[str, ...] = ("default",)
DOLMA_FIELDS = ("id", "text", "metadata")
DEFAULT_SPLIT = "train"

WORD_TO_TOKEN_MULTIPLIER = 1.35

TOPICS = [
    # Digital & Tech
    "software",
    "software_development",
    "electronics_and_hardware",
    "science_math_and_technology",
    # Media & Culture
    "entertainment",
    "games",
    "literature",
    "art_and_design",
    # Lifestyle
    "fashion_and_beauty",
    "food_and_dining",
    "home_and_hobbies",
    "sports_and_fitness",
    # People & Society
    "social_life",
    "education_and_jobs",
    "religion",
    "politics",
    # Money & Industry
    "finance_and_business",
    "industrial",
    "transportation",
    # Body & Place
    "health",
    "travel_and_tourism",
    "history_and_geography",
    # Sensitive
    "crime_and_law",
    "adult_content",
]

FORMATS = [
    # Long-form writing
    "academic_writing",
    "nonfiction_writing",
    "creative_writing",
    "personal_blog",
    # News & journalism
    "news_article",
    "news_org",
    "audio_transcript",
    # Reference & knowledge
    "knowledge_article",
    "faq",
    "q_a_forum",
    "tutorial",
    "documentation",
    # Structured / data
    "structured_data",
    "content_listing",
    "listicle",
    # Commercial
    "product_page",
    "user_review",
    "customer_support",
    "spam_ads",
    # Organizational
    "about_org",
    "about_pers",
    "legal_notices",
    # Social
    "comment_section",
    "truncated",
]

BIN_FULL = "full"
BIN_PARTIAL = "partial"
BIN_SPARSE = "sparse"
BIN_EMPTY = "empty"

TARGET_DOCS_PER_BIN = 100_000
TOKEN_FLOOR_PER_BIN = 500_000

WORKING_SAMPLE_TOKEN_FLOOR_PER_BIN = 1_000_000
WORKING_SAMPLE_SAMPLING_SEED = 42
WORKING_SAMPLE_MIN_TOKEN_COUNT = 512

PRECONDITIONER_SAMPLE_MIN_TOKENS = 512

__all__ = [
    "ALLOWED_DOLMA_DATASET_IDS",
    "ALLOWED_DOLMA_SUBSETS",
    "BIN_EMPTY",
    "BIN_FULL",
    "BIN_PARTIAL",
    "BIN_SPARSE",
    "DEFAULT_SPLIT",
    "DOLMA_6T_MIX_DATASET_ID",
    "DOLMA_DATASET_ID",
    "DOLMA_DOLMINO_MIX_1025_DATASET_ID",
    "DOLMA_FIELDS",
    "DOLMA_POOL_DATASET_ID",
    "FORMATS",
    "PRECONDITIONER_SAMPLE_MIN_TOKENS",
    "TARGET_DOCS_PER_BIN",
    "TOKEN_FLOOR_PER_BIN",
    "TOPICS",
    "WORD_TO_TOKEN_MULTIPLIER",
    "WORKING_SAMPLE_MIN_TOKEN_COUNT",
    "WORKING_SAMPLE_SAMPLING_SEED",
    "WORKING_SAMPLE_TOKEN_FLOOR_PER_BIN",
]
