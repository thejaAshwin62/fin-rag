"""
============================================================
ingest.py — Data Ingestion Pipeline
============================================================
Handles the complete ingestion flow:
1. Generate finance dataset
2. Chunk text with overlap
3. Generate embeddings via Gemini
4. Create Pinecone index (if needed)
5. Upsert vectors with metadata
============================================================
"""

import os
import time
import logging

from google import genai
from google.genai import types
from pinecone import Pinecone, ServerlessSpec

from config import (
    GOOGLE_API_KEY, PINECONE_API_KEY,
    EMBEDDING_MODEL, EMBEDDING_DIMENSION,
    PINECONE_INDEX_NAME, PINECONE_CLOUD, PINECONE_REGION,
    PINECONE_METRIC, PINECONE_NAMESPACE,
    CHUNK_SIZE, CHUNK_OVERLAP,
    UPSERT_BATCH_SIZE, EMBEDDING_BATCH_SIZE,
    DATA_DIR, DATASET_FILE,
)
from utils import (
    setup_logging, retry_with_backoff, hash_text,
    print_step, print_success, print_error, print_info,
    print_separator, Colors,
)

logger = logging.getLogger("rag_finance")


# ════════════════════════════════════════════════════════════
# SECTION 1: Finance Dataset Generation
# ════════════════════════════════════════════════════════════

# Professional finance content covering 12 domains
FINANCE_CONTENT = {
    "Investment Banking": [
        "Investment banking is a specialized segment of banking that helps organizations raise capital and provide advisory services for complex financial transactions. Investment banks act as intermediaries between securities issuers and investors, facilitating initial public offerings (IPOs), mergers and acquisitions (M&A), and debt issuances. The primary revenue streams include underwriting fees, advisory fees, and trading commissions.",
        "Mergers and acquisitions (M&A) represent a critical function of investment banking. The process involves identifying potential targets, conducting due diligence, structuring the deal, negotiating terms, and managing regulatory approvals. Valuation methodologies used include discounted cash flow (DCF) analysis, comparable company analysis, and precedent transaction analysis. Successful M&A deals require deep industry expertise and strong relationships with corporate executives.",
        "The IPO process involves multiple stages including selecting underwriters, filing registration statements with the SEC, conducting roadshows to generate investor interest, pricing the offering, and allocating shares. Investment banks use book-building methods to gauge demand and determine the optimal offering price. The underwriting spread, typically 3-7% of gross proceeds, compensates the investment bank for its risk and services.",
    ],
    "Risk Management": [
        "Financial risk management is the practice of identifying, analyzing, and mitigating uncertainty in investment decisions. The main categories of financial risk include market risk, credit risk, liquidity risk, operational risk, and systemic risk. Value at Risk (VaR) is a widely used statistical measure that quantifies the maximum potential loss over a specified time period at a given confidence level, typically 95% or 99%.",
        "Credit risk assessment involves evaluating the probability of default by a borrower. Key metrics include the probability of default (PD), loss given default (LGD), and exposure at default (EAD). Credit risk models such as the Merton model, CreditMetrics, and machine learning-based scoring systems help financial institutions quantify and manage their credit exposure across portfolios.",
        "Operational risk encompasses losses resulting from inadequate or failed internal processes, people, systems, or external events. The Basel III framework requires banks to maintain capital reserves against operational risk. Key risk indicators (KRIs), loss event databases, and scenario analysis are essential tools for managing operational risk effectively.",
    ],
    "Financial Derivatives": [
        "Financial derivatives are contracts whose value is derived from an underlying asset, index, or rate. The main types include options, futures, forwards, and swaps. Options give the holder the right but not the obligation to buy (call) or sell (put) an underlying asset at a predetermined strike price before expiration. The Black-Scholes model is the foundational pricing model for European options.",
        "Interest rate swaps are over-the-counter (OTC) derivatives where two parties exchange fixed-rate and floating-rate interest payments. These instruments are used extensively for hedging interest rate exposure and managing asset-liability mismatches. The notional principal amount is not exchanged; only the net interest differential is settled periodically. The swap market exceeds $400 trillion in notional value globally.",
        "Credit default swaps (CDS) are derivative contracts that transfer credit risk from one party to another. The buyer of a CDS makes periodic payments to the seller in exchange for protection against default by a reference entity. CDS played a significant role in the 2008 financial crisis, leading to increased regulatory oversight under the Dodd-Frank Act and mandatory central clearing requirements.",
    ],
    "Stock Market Analysis": [
        "Fundamental analysis evaluates securities by examining the intrinsic value of a company through financial statements, industry conditions, and macroeconomic factors. Key metrics include price-to-earnings (P/E) ratio, price-to-book (P/B) ratio, earnings per share (EPS), return on equity (ROE), and free cash flow (FCF). Analysts build detailed financial models to project future earnings and determine fair value estimates.",
        "Technical analysis studies price movements and trading volume patterns to forecast future price direction. Common tools include moving averages, relative strength index (RSI), MACD (Moving Average Convergence Divergence), Bollinger Bands, and Fibonacci retracement levels. Chart patterns such as head and shoulders, double tops, and triangles help traders identify potential trend reversals and continuation patterns.",
        "Quantitative analysis applies mathematical and statistical models to financial markets. Factor models like the Fama-French three-factor model and Carhart four-factor model decompose returns into systematic risk factors. High-frequency trading (HFT) algorithms execute thousands of trades per second, exploiting microscopic price inefficiencies. Machine learning techniques including random forests and neural networks are increasingly used for alpha generation.",
    ],
    "Mutual Funds": [
        "Mutual funds pool money from multiple investors to invest in a diversified portfolio of stocks, bonds, or other securities managed by professional fund managers. Key types include equity funds, bond funds, money market funds, index funds, and balanced funds. The expense ratio, which typically ranges from 0.03% for index funds to over 1.5% for actively managed funds, directly impacts investor returns over time.",
        "Net Asset Value (NAV) is calculated by dividing the total value of all securities in the portfolio minus liabilities by the number of outstanding shares. NAV is computed at the end of each trading day. Investors can buy or redeem mutual fund shares at the NAV price. Load funds charge sales commissions (front-end or back-end loads), while no-load funds do not charge sales commissions.",
        "Exchange-Traded Funds (ETFs) combine features of mutual funds and stocks. They trade on stock exchanges throughout the day at market prices, offer tax efficiency through the creation/redemption mechanism, and typically have lower expense ratios than comparable mutual funds. Smart beta ETFs use alternative index construction rules based on factors like value, momentum, quality, and low volatility.",
    ],
    "Credit Scoring": [
        "Credit scoring models assess the creditworthiness of individuals and businesses using statistical algorithms. FICO scores, ranging from 300 to 850, consider payment history (35%), amounts owed (30%), length of credit history (15%), new credit (10%), and credit mix (10%). Advanced machine learning models incorporating alternative data sources like utility payments and rental history are expanding credit access to underbanked populations.",
        "Basel III regulatory framework requires banks to maintain minimum capital adequacy ratios based on risk-weighted assets. The Common Equity Tier 1 (CET1) ratio must be at least 4.5%, with additional capital conservation and countercyclical buffers. Internal Ratings-Based (IRB) approaches allow qualified banks to use their own credit risk models for calculating regulatory capital requirements.",
        "Alternative credit scoring leverages non-traditional data sources including mobile phone usage patterns, social media activity, e-commerce transaction history, and psychometric assessments. Fintech companies like Upstart and ZestFinance use machine learning algorithms to analyze hundreds of variables, potentially offering more accurate risk assessments for thin-file borrowers who lack conventional credit histories.",
    ],
    "Fraud Detection": [
        "Financial fraud detection systems use machine learning algorithms to identify suspicious transactions in real-time. Common techniques include anomaly detection using isolation forests, supervised classification using gradient boosting machines (XGBoost, LightGBM), and deep learning approaches using autoencoders and recurrent neural networks. These systems process millions of transactions daily, flagging potentially fraudulent activities for investigation.",
        "Anti-money laundering (AML) compliance requires financial institutions to implement Know Your Customer (KYC) procedures, transaction monitoring systems, and suspicious activity reporting (SAR). Advanced AML solutions use network analysis to detect complex laundering schemes involving multiple accounts, shell companies, and cross-border transactions. Regulatory fines for AML violations can exceed billions of dollars.",
        "Identity theft and account takeover fraud have increased significantly with digital banking adoption. Multi-factor authentication (MFA), behavioral biometrics, device fingerprinting, and real-time transaction monitoring are essential defense layers. Machine learning models trained on historical fraud patterns achieve detection rates above 95% while maintaining false positive rates below 0.1%.",
    ],
    "Wealth Management": [
        "Wealth management is a comprehensive advisory service that combines financial planning, investment management, tax optimization, and estate planning for high-net-worth individuals (HNWIs). The typical wealth management relationship begins with understanding the client's financial goals, risk tolerance, time horizon, and liquidity needs. Assets under management (AUM) fees typically range from 0.5% to 1.5% annually.",
        "Asset allocation is the strategic distribution of investment capital across different asset classes including equities, fixed income, real estate, commodities, and alternative investments. Modern Portfolio Theory (MPT), developed by Harry Markowitz, demonstrates that diversification can optimize the risk-return tradeoff. Strategic asset allocation establishes long-term targets, while tactical allocation makes short-term adjustments based on market conditions.",
        "Estate planning involves structuring the transfer of wealth to beneficiaries in a tax-efficient manner. Key instruments include revocable and irrevocable trusts, family limited partnerships (FLPs), grantor retained annuity trusts (GRATs), and charitable remainder trusts (CRTs). The current federal estate tax exemption is $13.61 million per individual, with a top marginal rate of 40% on amounts exceeding the exemption.",
    ],
    "Portfolio Optimization": [
        "Portfolio optimization uses mathematical frameworks to select the best portfolio allocation given a set of constraints. The mean-variance optimization framework, introduced by Markowitz, constructs the efficient frontier — the set of portfolios offering the highest expected return for each level of risk. Inputs include expected returns, volatilities, and correlations for all assets under consideration.",
        "The Capital Asset Pricing Model (CAPM) establishes a linear relationship between expected return and systematic risk (beta). The Security Market Line (SML) plots this relationship, with assets above the line considered undervalued and those below considered overvalued. While CAPM has theoretical elegance, empirical evidence shows that additional factors like size, value, and momentum explain cross-sectional return variations.",
        "Risk parity is an allocation approach that equalizes the risk contribution of each asset class rather than allocating by capital. This methodology typically results in higher allocations to bonds and lower allocations to equities compared to traditional 60/40 portfolios. Leveraged risk parity strategies can enhance returns while maintaining the balanced risk profile, though they introduce leverage risk.",
    ],
    "Corporate Finance": [
        "Corporate finance focuses on how corporations manage their capital structure, funding sources, and investment decisions to maximize shareholder value. The weighted average cost of capital (WACC) serves as the discount rate for evaluating new investment projects. Capital budgeting techniques including net present value (NPV), internal rate of return (IRR), and payback period help managers allocate resources to value-creating projects.",
        "Capital structure theory examines the optimal mix of debt and equity financing. The Modigliani-Miller theorem states that in perfect markets, capital structure is irrelevant to firm value. However, in practice, the tax deductibility of interest payments creates a tax shield that favors debt financing, while excessive leverage increases financial distress costs and bankruptcy risk. The trade-off theory balances these opposing forces.",
        "Working capital management involves managing short-term assets and liabilities to ensure operational liquidity. Key components include accounts receivable management, inventory optimization, and accounts payable strategies. The cash conversion cycle (CCC) measures the time between cash outflow for raw materials and cash inflow from sales. Efficient working capital management improves free cash flow and reduces the need for external financing.",
    ],
    "Financial Compliance": [
        "Financial regulatory compliance encompasses adherence to laws, regulations, and guidelines governing financial institutions. Key frameworks include the Dodd-Frank Wall Street Reform Act, Sarbanes-Oxley Act (SOX), Markets in Financial Instruments Directive (MiFID II), and General Data Protection Regulation (GDPR). Compliance programs require robust internal controls, regular auditing, employee training, and reporting mechanisms.",
        "The Dodd-Frank Act, enacted in 2010 in response to the financial crisis, introduced sweeping reforms including the Volcker Rule (restricting proprietary trading), creation of the Consumer Financial Protection Bureau (CFPB), mandatory central clearing for standardized derivatives, and enhanced capital and liquidity requirements for systemically important financial institutions (SIFIs).",
        "RegTech (Regulatory Technology) solutions leverage artificial intelligence, machine learning, and blockchain to automate compliance processes. These technologies enable real-time transaction monitoring, automated regulatory reporting, digital identity verification, and predictive compliance risk assessment. The global RegTech market is projected to exceed $30 billion, driven by increasing regulatory complexity and enforcement actions.",
    ],
    "FinTech Systems": [
        "Financial Technology (FinTech) encompasses technological innovations that compete with or enhance traditional financial services. Key areas include digital payments, peer-to-peer lending, robo-advisory, blockchain and cryptocurrency, insurtech, and open banking. APIs enable seamless integration between financial service providers, creating an ecosystem of interconnected services that improve customer experience and operational efficiency.",
        "Blockchain technology provides a decentralized, immutable ledger for recording financial transactions. Smart contracts on platforms like Ethereum enable programmable financial agreements that execute automatically when conditions are met. Decentralized Finance (DeFi) protocols offer lending, borrowing, and trading services without traditional intermediaries, though regulatory frameworks are still evolving to address associated risks.",
        "Open banking regulations like PSD2 in Europe require banks to share customer financial data with authorized third-party providers through secure APIs. This enables innovative services such as account aggregation, automated savings tools, personalized financial advice, and streamlined lending processes. Banking-as-a-Service (BaaS) platforms allow non-financial companies to embed banking services into their products.",
    ],
}


def generate_finance_dataset() -> str:
    """
    Generate a comprehensive finance dataset from predefined content
    and save it to data/finance_dataset.txt.
    
    Returns:
        str: The complete dataset text.
    """
    logger.info("Generating finance dataset...")
    os.makedirs(DATA_DIR, exist_ok=True)

    lines = []
    lines.append("=" * 70)
    lines.append("FINANCE KNOWLEDGE BASE — RAG CHATBOT DATASET")
    lines.append("=" * 70)
    lines.append("")

    for topic, paragraphs in FINANCE_CONTENT.items():
        lines.append(f"\n{'─' * 50}")
        lines.append(f"TOPIC: {topic.upper()}")
        lines.append(f"{'─' * 50}\n")
        for para in paragraphs:
            lines.append(para)
            lines.append("")  # blank line between paragraphs

    full_text = "\n".join(lines)

    # Save dataset to file
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        f.write(full_text)

    word_count = len(full_text.split())
    logger.info("Dataset generated: %d words, saved to %s", word_count, DATASET_FILE)
    print_success(f"Dataset generated: {word_count:,} words → {DATASET_FILE}")
    return full_text


# ════════════════════════════════════════════════════════════
# SECTION 2: Text Chunking
# ════════════════════════════════════════════════════════════

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split text into overlapping chunks with metadata.
    
    Args:
        text: Full text to chunk.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.
    
    Returns:
        List of dicts with keys: text, chunk_index, topic, source.
    """
    chunks = []
    # Identify topic sections for metadata
    current_topic = "General"

    # Split into paragraphs first for cleaner chunking
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Rebuild as continuous text for sliding-window chunking
    continuous = " ".join(paragraphs)

    start = 0
    chunk_index = 0

    while start < len(continuous):
        end = start + chunk_size

        # Try to break at a sentence boundary (period + space)
        if end < len(continuous):
            # Look backwards from 'end' for a sentence boundary
            boundary = continuous.rfind(". ", start, end)
            if boundary > start:
                end = boundary + 1  # include the period

        chunk_text_str = continuous[start:end].strip()

        if chunk_text_str:
            # Detect topic from content keywords
            detected_topic = _detect_topic(chunk_text_str)

            chunks.append({
                "text": chunk_text_str,
                "chunk_index": chunk_index,
                "topic": detected_topic,
                "source": "finance_dataset.txt",
            })
            chunk_index += 1

        start = end - overlap if end < len(continuous) else len(continuous)

    logger.info("Text chunked into %d chunks (size=%d, overlap=%d)", len(chunks), chunk_size, overlap)
    print_success(f"Text split into {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks


def _detect_topic(text: str) -> str:
    """Detect the finance topic of a chunk based on keyword matching."""
    topic_keywords = {
        "Investment Banking": ["investment bank", "IPO", "underwriting", "M&A", "merger"],
        "Risk Management": ["risk management", "VaR", "credit risk", "operational risk", "Basel"],
        "Financial Derivatives": ["derivative", "option", "swap", "futures", "Black-Scholes"],
        "Stock Market Analysis": ["stock market", "technical analysis", "fundamental analysis", "P/E ratio", "MACD"],
        "Mutual Funds": ["mutual fund", "NAV", "ETF", "expense ratio", "index fund"],
        "Credit Scoring": ["credit scor", "FICO", "creditworth", "Basel III", "CET1"],
        "Fraud Detection": ["fraud", "anti-money laundering", "AML", "KYC", "suspicious"],
        "Wealth Management": ["wealth management", "estate planning", "asset allocation", "HNWI", "trust"],
        "Portfolio Optimization": ["portfolio optim", "Markowitz", "efficient frontier", "CAPM", "risk parity"],
        "Corporate Finance": ["corporate finance", "WACC", "capital structure", "NPV", "working capital"],
        "Financial Compliance": ["compliance", "Dodd-Frank", "SOX", "RegTech", "regulatory"],
        "FinTech Systems": ["fintech", "blockchain", "DeFi", "open banking", "smart contract"],
    }

    text_lower = text.lower()
    for topic, keywords in topic_keywords.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                return topic

    return "General Finance"


# ════════════════════════════════════════════════════════════
# SECTION 3: Embedding Generation
# ════════════════════════════════════════════════════════════

@retry_with_backoff(max_retries=3)
def _embed_single(client: genai.Client, text: str) -> list[float]:
    """Embed a single text string using Gemini embedding model (with retry)."""
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSION,
        ),
    )
    return result.embeddings[0].values


def generate_embeddings(chunks: list[dict]) -> list[dict]:
    """
    Generate embeddings for all chunks using Gemini embedding model.
    Embeds each chunk individually to guarantee one vector per chunk.
    
    Args:
        chunks: List of chunk dicts (must have 'text' key).
    
    Returns:
        List of dicts with added 'embedding' key containing float vectors.
    """
    logger.info("Generating embeddings for %d chunks...", len(chunks))
    print_info(f"Generating embeddings for {len(chunks)} chunks (model: {EMBEDDING_MODEL})...")

    client = genai.Client(api_key=GOOGLE_API_KEY)

    # Embed each chunk individually for reliable 1:1 mapping
    for i, chunk in enumerate(chunks):
        embedding = _embed_single(client, chunk["text"])
        chunk["embedding"] = embedding

        # Progress update every 10 chunks
        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"    {Colors.DIM}Embedded {i + 1}/{len(chunks)} chunks...{Colors.RESET}")

    logger.info("Embeddings generated successfully (dim=%d)", EMBEDDING_DIMENSION)
    print_success(f"All embeddings generated (dimension={EMBEDDING_DIMENSION})")
    return chunks


# ════════════════════════════════════════════════════════════
# SECTION 4: Pinecone Index Management
# ════════════════════════════════════════════════════════════

def init_pinecone_index() -> object:
    """
    Initialize Pinecone client and create the index if it doesn't exist.
    
    Returns:
        Pinecone Index object ready for upsert/query operations.
    """
    logger.info("Initializing Pinecone (index=%s)...", PINECONE_INDEX_NAME)
    print_info(f"Connecting to Pinecone (index: {PINECONE_INDEX_NAME})...")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if index already exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        print_info(f"Creating new index '{PINECONE_INDEX_NAME}' ({PINECONE_CLOUD}/{PINECONE_REGION})...")
        logger.info("Creating Pinecone index: %s", PINECONE_INDEX_NAME)

        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )

        # Wait for index to be ready
        print_info("Waiting for index to be ready...")
        while not pc.describe_index(PINECONE_INDEX_NAME).status.get("ready", False):
            time.sleep(2)

        print_success(f"Index '{PINECONE_INDEX_NAME}' created and ready!")
        logger.info("Pinecone index created successfully")
    else:
        print_success(f"Index '{PINECONE_INDEX_NAME}' already exists")
        logger.info("Pinecone index already exists")

    # Return index handle
    index = pc.Index(PINECONE_INDEX_NAME)
    return index


# ════════════════════════════════════════════════════════════
# SECTION 5: Vector Upsert to Pinecone
# ════════════════════════════════════════════════════════════

def upsert_to_pinecone(index, chunks: list[dict]):
    """
    Upsert embedding vectors with metadata into Pinecone.
    Uses MD5 hash of text as vector ID to prevent duplicates.
    
    Args:
        index: Pinecone Index object.
        chunks: List of chunk dicts (must have 'text', 'embedding', 'topic', 'chunk_index').
    """
    logger.info("Upserting %d vectors to Pinecone...", len(chunks))
    print_info(f"Upserting {len(chunks)} vectors to Pinecone...")

    total_upserted = 0

    for i in range(0, len(chunks), UPSERT_BATCH_SIZE):
        batch = chunks[i : i + UPSERT_BATCH_SIZE]

        vectors = []
        for chunk in batch:
            # Use MD5 hash as vector ID — prevents duplicates on re-ingestion
            vec_id = hash_text(chunk["text"])

            vectors.append({
                "id": vec_id,
                "values": chunk["embedding"],
                "metadata": {
                    "text": chunk["text"],
                    "topic": chunk["topic"],
                    "chunk_index": chunk["chunk_index"],
                    "source": chunk["source"],
                },
            })

        index.upsert(vectors=vectors, namespace=PINECONE_NAMESPACE)
        total_upserted += len(vectors)
        print(f"    {Colors.DIM}Upserted {total_upserted}/{len(chunks)} vectors...{Colors.RESET}")

    logger.info("Successfully upserted %d vectors", total_upserted)
    print_success(f"All {total_upserted} vectors stored in Pinecone!")


# ════════════════════════════════════════════════════════════
# SECTION 6: Orchestrator
# ════════════════════════════════════════════════════════════

def run_ingestion():
    """
    Run the complete data ingestion pipeline:
    1. Generate finance dataset
    2. Chunk text
    3. Generate embeddings
    4. Create/connect Pinecone index
    5. Upsert vectors
    """
    print_separator("═", 60, Colors.MAGENTA)
    print(f"  {Colors.BOLD}{Colors.MAGENTA}📥 DATA INGESTION PIPELINE{Colors.RESET}")
    print_separator("═", 60, Colors.MAGENTA)
    print()

    try:
        # Step 1: Generate dataset
        print_step(1, "Generating finance dataset...")
        dataset_text = generate_finance_dataset()
        print()

        # Step 2: Chunk text
        print_step(2, "Chunking text into overlapping segments...")
        chunks = chunk_text(dataset_text)
        print()

        # Step 3: Generate embeddings
        print_step(3, "Generating embeddings via Gemini...")
        chunks = generate_embeddings(chunks)
        print()

        # Step 4: Initialize Pinecone
        print_step(4, "Initializing Pinecone index...")
        index = init_pinecone_index()
        print()

        # Step 5: Upsert vectors
        print_step(5, "Upserting vectors to Pinecone...")
        upsert_to_pinecone(index, chunks)
        print()

        print_separator("═", 60, Colors.GREEN)
        print(f"  {Colors.BOLD}{Colors.GREEN}✅ INGESTION COMPLETE!{Colors.RESET}")
        print(f"  {Colors.DIM}Chunks: {len(chunks)} | Dimension: {EMBEDDING_DIMENSION} | "
              f"Index: {PINECONE_INDEX_NAME}{Colors.RESET}")
        print_separator("═", 60, Colors.GREEN)
        print()

    except Exception as e:
        logger.error("Ingestion pipeline failed: %s", str(e), exc_info=True)
        print_error(f"Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    setup_logging()
    from config import validate_env
    validate_env()
    run_ingestion()
