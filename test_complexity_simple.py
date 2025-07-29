#!/usr/bin/env python3
"""
Simple test for enhanced complexity analyzer integration
"""

import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.complexity_detector import get_complexity_detector
from models.query_features import QueryComplexity, KeywordExtraction


async def test_complexity_detector():
    """Test the enhanced complexity detector."""
    print("Testing Enhanced Complexity Detector...")
    
    detector = get_complexity_detector()
    
    # Test queries with different complexity levels
    test_cases = [
        ("find user", QueryComplexity.SIMPLE),
        ("analyze the connection between authentication service and user management", QueryComplexity.MODERATE),
        ("implement a comprehensive architectural framework that integrates multiple microservices", QueryComplexity.COMPLEX),
    ]
    
    for query, expected_range in test_cases:
        print(f"\nAnalyzing: '{query}'")
        
        # Create some mock keywords
        keywords = KeywordExtraction(
            entity_names=['user', 'service'] if 'user' in query or 'service' in query else [],
            technical_terms=['authentication', 'architectural'] if any(term in query for term in ['authentication', 'architectural']) else [],
            concept_terms=['connection', 'framework'] if any(term in query for term in ['connection', 'framework']) else []
        )
        
        profile = await detector.analyze_complexity(query, keywords=keywords)
        
        print(f"  Complexity: {profile.overall_complexity.value}")
        print(f"  Score: {profile.complexity_score:.3f}")
        print(f"  Confidence: {profile.confidence:.3f}")
        print(f"  Lexical: {profile.lexical_complexity:.3f}")
        print(f"  Syntactic: {profile.syntactic_complexity:.3f}")
        print(f"  Semantic: {profile.semantic_complexity:.3f}")
        print(f"  Analysis Quality: {profile.analysis_quality:.3f}")
        
        # Check if the complexity is reasonable
        assert profile.complexity_score >= 0.0 and profile.complexity_score <= 1.0
        assert profile.confidence >= 0.0 and profile.confidence <= 1.0
        print("  ✓ Passed validation")
    
    print("\n✅ All complexity detector tests passed!")


async def main():
    """Main test function."""
    try:
        await test_complexity_detector()
        print("\n🎉 All tests completed successfully!")
        
        # Get and display stats
        detector = get_complexity_detector()
        stats = detector.get_analysis_statistics()
        print(f"\nAnalysis Statistics:")
        print(f"  Analyses performed: {stats['analyses_performed']}")
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
        print(f"  Average analysis time: {stats['average_analysis_time_ms']:.2f}ms")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)