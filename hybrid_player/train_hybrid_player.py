
import os
import sys
import logging

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from hybrid_player import HybridPlayerConfig, HybridPlayerTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_data_paths(config):
    issues = []
    
    # Check CRD3 paths
    crd3_path = config.data.crd3_base_path
    if not os.path.exists(crd3_path):
        issues.append(f"CRD3 path not found: {crd3_path}")
    else:
        # Check for aligned_data subdirectory
        aligned_path = os.path.join(crd3_path, "aligned_data")
        if not os.path.exists(aligned_path):
            issues.append(f"CRD3 aligned_data not found: {aligned_path}")
    
    # Check LIGHT paths
    light_path = config.data.light_base_path
    if not os.path.exists(light_path):
        issues.append(f"LIGHT path not found: {light_path}")
    else:
        light_files = ['light_data.pkl', 'light_unseen_data.pkl']
        found_files = [f for f in light_files if os.path.exists(os.path.join(light_path, f))]
        if not found_files:
            issues.append(f"No LIGHT data files found in: {light_path}")
    
    return issues

def main():
    """Main training function"""
    try:
        logger.info("Starting Hybrid Player Training Pipeline")
        
        # Initialize configuration
        config = HybridPlayerConfig()
        
        # Validate data paths
        logger.info("Validating data paths...")
        path_issues = validate_data_paths(config)
        if path_issues:
            logger.warning("Data path issues found:")
            for issue in path_issues:
                logger.warning(f"  - {issue}")
            logger.warning("Training may fail if data is not accessible")
        else:
            logger.info(" All data paths validated successfully")
        
        # Initialize trainer
        trainer = HybridPlayerTrainer(config)
        
        # Train all components
        logger.info("Beginning training process...")
        train_df, val_df, test_df = trainer.train_all()
        
        logger.info(" Training completed successfully!")
        logger.info(f"Final dataset sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.error("Please check that:")
        logger.error("1. Your data files are in the correct locations")
        logger.error("2. The file structures match the expected formats")
        logger.error("3. You have read permissions for the data files")
        raise

if __name__ == "__main__":
    main()