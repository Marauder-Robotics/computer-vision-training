#!/bin/bash
# Marauder CV Project - Data Pipeline Testing Script
# Tests data downloaders, dataset organizer, and DO bucket structure
# Usage: ./scripts/test_data_pipeline.sh [--full] [--quick]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test modes
FULL_TEST=false
QUICK_TEST=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            FULL_TEST=true
            shift
            ;;
        --quick)
            QUICK_TEST=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--full] [--quick]"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Marauder CV - Data Pipeline Tests"
echo "========================================"
echo ""

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0

# Helper function for test reporting
test_result() {
    local test_name=$1
    local result=$2
    
    if [ $result -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $test_name"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗${NC} $test_name"
        ((TESTS_FAILED++))
    fi
}

echo "=== Test 1: Environment Check ==="
echo ""

# Check Python version
echo -n "Checking Python version... "
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
if [[ $PYTHON_VERSION > "3.8" ]]; then
    echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗${NC} Python $PYTHON_VERSION (need 3.8+)"
    ((TESTS_FAILED++))
fi

# Check required packages
echo -n "Checking required packages... "
python -c "import yaml, PIL, tqdm, requests" 2>/dev/null
test_result "Required packages" $?

echo ""
echo "=== Test 2: DO Bucket Accessibility ==="
echo ""

# Check if DO bucket is mounted
echo -n "Checking DO bucket mount... "
if [ -d "/datasets/marauder-do-bucket" ]; then
    echo -e "${GREEN}✓${NC} Mounted at /datasets/marauder-do-bucket"
    ((TESTS_PASSED++))
else
    echo -e "${RED}✗${NC} DO bucket not found at /datasets/marauder-do-bucket"
    ((TESTS_FAILED++))
    echo "Skipping remaining DO bucket tests..."
fi

# Check DO bucket structure if mounted
if [ -d "/datasets/marauder-do-bucket" ]; then
    echo -n "Checking images directory... "
    [ -d "/datasets/marauder-do-bucket/images" ]
    test_result "images/ directory" $?
    
    echo -n "Checking training directory... "
    [ -d "/datasets/marauder-do-bucket/training" ]
    test_result "training/ directory" $?
    
    echo -n "Checking fathomnet images... "
    [ -d "/datasets/marauder-do-bucket/images/fathomnet" ]
    test_result "images/fathomnet/ directory" $?
    
    echo -n "Checking deepfish images... "
    [ -d "/datasets/marauder-do-bucket/images/deepfish" ]
    test_result "images/deepfish/ directory" $?
    
    echo -n "Checking marauder images... "
    [ -d "/datasets/marauder-do-bucket/images/marauder" ]
    test_result "images/marauder/ directory" $?
fi

echo ""
echo "=== Test 3: Project Structure ==="
echo ""

# Check project directories
echo -n "Checking config directory... "
[ -d "config" ]
test_result "config/ directory" $?

echo -n "Checking data directory... "
[ -d "data" ]
test_result "data/ directory" $?

echo -n "Checking training directory... "
[ -d "training" ]
test_result "training/ directory" $?

echo -n "Checking scripts directory... "
[ -d "scripts" ]
test_result "scripts/ directory" $?

# Check config files
echo -n "Checking species_mapping.yaml... "
[ -f "config/species_mapping.yaml" ]
test_result "species_mapping.yaml" $?

echo -n "Checking training_config.yaml... "
[ -f "config/training_config.yaml" ]
test_result "training_config.yaml" $?

echo ""
echo "=== Test 4: Data Acquisition Scripts ==="
echo ""

# Check if data acquisition scripts exist
echo -n "Checking FathomNet downloader... "
[ -f "data/acquisition/fathomnet_downloader.py" ]
test_result "fathomnet_downloader.py" $?

echo -n "Checking DeepFish downloader... "
[ -f "data/acquisition/deepfish_downloader.py" ]
test_result "deepfish_downloader.py" $?

echo -n "Checking dataset organizer... "
[ -f "data/acquisition/dataset_organizer.py" ]
test_result "dataset_organizer.py" $?

# Test script syntax (dry run)
if [ "$QUICK_TEST" = false ]; then
    echo ""
    echo -n "Testing FathomNet downloader syntax... "
    python data/acquisition/fathomnet_downloader.py --help > /dev/null 2>&1
    test_result "FathomNet syntax check" $?
    
    echo -n "Testing DeepFish downloader syntax... "
    python data/acquisition/deepfish_downloader.py --help > /dev/null 2>&1
    test_result "DeepFish syntax check" $?
    
    echo -n "Testing dataset organizer syntax... "
    python data/acquisition/dataset_organizer.py --help > /dev/null 2>&1
    test_result "Dataset organizer syntax check" $?
fi

echo ""
echo "=== Test 5: Preprocessing Scripts ==="
echo ""

echo -n "Checking hybrid preprocessor... "
[ -f "data/preprocessing/hybrid_preprocessor.py" ]
test_result "hybrid_preprocessor.py" $?

if [ "$QUICK_TEST" = false ]; then
    echo -n "Testing preprocessor syntax... "
    python data/preprocessing/hybrid_preprocessor.py --help > /dev/null 2>&1
    test_result "Preprocessor syntax check" $?
fi

echo ""
echo "=== Test 6: Training Scripts ==="
echo ""

# Check training scripts
echo -n "Checking SSL pretrain... "
[ -f "training/1_ssl_pretrain.py" ]
test_result "1_ssl_pretrain.py" $?

echo -n "Checking baseline YOLO... "
[ -f "training/2_baseline_yolo.py" ]
test_result "2_baseline_yolo.py" $?

echo -n "Checking active learning... "
[ -f "training/3_active_learning.py" ]
test_result "3_active_learning.py" $?

echo -n "Checking critical species... "
[ -f "training/4_critical_species.py" ]
test_result "4_critical_species.py" $?

echo ""
echo "=== Test 7: Utility Scripts ==="
echo ""

echo -n "Checking TrainingLogger... "
[ -f "utils/training_logger.py" ]
test_result "training_logger.py" $?

echo -n "Checking CheckpointManager... "
[ -f "utils/checkpoint_manager.py" ]
test_result "checkpoint_manager.py" $?

echo ""
echo "=== Test 8: Dataset Organizer (Scan Mode) ==="
echo ""

if [ "$FULL_TEST" = true ] && [ -d "/datasets/marauder-do-bucket/images" ]; then
    echo "Running dataset organizer in scan-only mode..."
    python data/acquisition/dataset_organizer.py \
        --input-dirs /datasets/marauder-do-bucket/images/fathomnet \
                     /datasets/marauder-do-bucket/images/deepfish \
                     /datasets/marauder-do-bucket/images/marauder \
        --output /tmp/test_organized \
        --config config/species_mapping.yaml \
        --scan-only
    
    test_result "Dataset organizer scan" $?
else
    echo -e "${YELLOW}⊘${NC} Skipping full test (use --full flag)"
fi

echo ""
echo "========================================"
echo "Test Results Summary"
echo "========================================"
echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo "Please review the output above and fix any issues."
    exit 1
fi
