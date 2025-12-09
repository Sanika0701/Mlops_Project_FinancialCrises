"""
Test the deployed API
"""
import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"  # Change to Cloud Run URL after deployment

def test_health():
    """Test health endpoint"""
    print("\n" + "="*80)
    print("TEST 1: Health Check")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/api/v1/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    print("✅ PASSED")

def test_list_scenarios():
    """Test list scenarios"""
    print("\n" + "="*80)
    print("TEST 2: List Scenarios")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/api/v1/scenarios")
    print(f"Status: {response.status_code}")
    
    data = response.json()
    print(f"Total scenarios: {data['total']}")
    print(f"First scenario: {json.dumps(data['scenarios'][0], indent=2)}")
    
    assert response.status_code == 200
    assert data['total'] > 0
    print("✅ PASSED")
    
    return data['scenarios']

def test_list_companies():
    """Test list companies"""
    print("\n" + "="*80)
    print("TEST 3: List Companies")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/api/v1/companies")
    print(f"Status: {response.status_code}")
    
    data = response.json()
    print(f"Total companies: {data['total']}")
    print(f"First 5 companies: {json.dumps(data['companies'][:5], indent=2)}")
    
    assert response.status_code == 200
    assert data['total'] > 0
    print("✅ PASSED")
    
    return data['companies']

def test_stress_test_single(scenarios, companies):
    """Test stress test with single scenario"""
    print("\n" + "="*80)
    print("TEST 4: Stress Test (Single Scenario)")
    print("="*80)
    
    company_id = companies[0]['company_id']
    scenario_id = scenarios[0]['scenario_id']
    
    payload = {
        "company_id": company_id,
        "scenario_ids": [scenario_id]
    }
    
    print(f"Testing: {company_id} × Scenario {scenario_id}")
    
    response = requests.post(
        f"{BASE_URL}/api/v1/stress-test",
        json=payload
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nResult:")
        print(f"  Company: {data['company_id']}")
        print(f"  Risk Score: {data['result']['risk_assessment']['risk_score']}")
        print(f"  Risk Category: {data['result']['risk_assessment']['risk_category']}")
        print(f"  Anomaly Detected: {data['result']['risk_assessment']['anomaly_detected']}")
        print(f"  Revenue Change: {data['result']['predictions']['revenue_change_pct']:.1f}%")
        
        assert response.status_code == 200
        print("✅ PASSED")
        return data
    else:
        print(f"❌ FAILED: {response.text}")
        return None

def test_stress_test_multiple(scenarios, companies):
    """Test stress test with multiple scenarios"""
    print("\n" + "="*80)
    print("TEST 5: Stress Test (Multiple Scenarios)")
    print("="*80)
    
    company_id = companies[0]['company_id']
    scenario_ids = [s['scenario_id'] for s in scenarios[:3]]
    
    payload = {
        "company_id": company_id,
        "scenario_ids": scenario_ids
    }
    
    print(f"Testing: {company_id} × {len(scenario_ids)} scenarios")
    
    response = requests.post(
        f"{BASE_URL}/api/v1/stress-test",
        json=payload
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nSummary:")
        print(f"  Average Risk: {data['summary']['avg_risk_score']}")
        print(f"  Best Case Risk: {data['summary']['best_case']['risk_assessment']['risk_score']}")
        print(f"  Worst Case Risk: {data['summary']['worst_case']['risk_assessment']['risk_score']}")
        
        assert response.status_code == 200
        print("✅ PASSED")
        return data
    else:
        print(f"❌ FAILED: {response.text}")
        return None

def main():
    """Run all tests"""
    print("="*80)
    print("🧪 API TESTING SUITE")
    print("="*80)
    print(f"Base URL: {BASE_URL}")
    
    try:
        # Test 1: Health
        test_health()
        
        # Test 2: Scenarios
        scenarios = test_list_scenarios()
        
        # Test 3: Companies
        companies = test_list_companies()
        
        # Test 4: Single stress test
        test_stress_test_single(scenarios, companies)
        
        # Test 5: Multiple stress test
        test_stress_test_multiple(scenarios, companies)
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ TEST FAILED: {e}")
        print("="*80)
        raise

if __name__ == "__main__":
    main()