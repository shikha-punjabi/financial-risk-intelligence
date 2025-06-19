from src.models.free_sentiment_analyzer import FreeSentimentAnalyzer
from src.models.free_llm_analyzer import FreeLLMAnalyzer
import pandas as pd

def test_ai_models():
    """Test AI models with sample data"""
    # Sample financial news for testing
    sample_news = [
        {
            'title': 'Apple reports record quarterly earnings, beats expectations',
            'description': 'Apple Inc. announced strong quarterly results with revenue growth of 15% year-over-year, exceeding analyst expectations.',
            'source': 'Financial Times'
        },
        {
            'title': 'Federal Reserve raises interest rates by 0.25%',
            'description': 'The Federal Reserve announced a quarter-point rate hike citing persistent inflation concerns and strong labor market.',
            'source': 'Reuters'
        },
        {
            'title': 'Banking sector faces potential credit losses amid economic uncertainty',
            'description': 'Major banks are increasing provisions for credit losses as economic indicators show signs of slowdown and rising defaults.',
            'source': 'Wall Street Journal'
        }
    ]
    sample_df = pd.DataFrame(sample_news)
    # Test sentiment analyzer
    print("Testing Free Sentiment Analyzer...")
    try:
        sentiment_analyzer = FreeSentimentAnalyzer()
        analyzed_news = sentiment_analyzer.analyze_news_batch(sample_df)
        print("✅ Sentiment Analysis Results:")
        for idx, row in analyzed_news.iterrows():
            print(f"   Article {idx + 1}: {row['title'][:50]}...")
            print(f"   Sentiment: {row['sentiment_score']:.3f}")
            print(f"   Risk Level: {row['risk_level']}")
            print(f"   Confidence: {row['sentiment_confidence']:.3f}")
            print()
        # Market sentiment summary
        summary = sentiment_analyzer.get_market_sentiment_summary(analyzed_news)
        print(f"Market Mood: {summary['market_mood']}")
        print(f"Risk Assessment: {summary['risk_assessment']}")
        print()
    except Exception as e:
        print(f"❌ Sentiment analysis failed: {e}")
        return False
    # Test LLM analyzer
    print("Testing Free LLM Analyzer...")
    try:
        llm_analyzer = FreeLLMAnalyzer()
        # Test with first article
        sample_text = f"{sample_news[0]['title']} {sample_news[0]['description']}"
        # Risk analysis
        risk_analysis = llm_analyzer.analyze_text_with_qa(sample_text, 'risk_analysis')
        print("✅ Risk Analysis Results:")
        if 'answers' in risk_analysis:
            for question, answer in risk_analysis['answers'].items():
                print(f"   Q: {question}")
                print(f"   A: {answer['answer']} (confidence: {answer['confidence']:.3f})")
                print()
        # Entity extraction
        entities = llm_analyzer.extract_key_entities(sample_text)
        print("✅ Entity Extraction Results:")
        print(entities)
        # Financial classification
        classification = llm_analyzer.classify_financial_content(sample_text)
        print("✅ Financial Classification Results:")
        print(classification)
        print()
    except Exception as e:
        print(f"❌ LLM analysis failed: {e}")
        return False
    print("✅ All AI model tests passed!")
    return True

if __name__ == "__main__":
    test_ai_models()
