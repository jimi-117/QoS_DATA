from icrawler.builtin import BingImageCrawler
import os
from pathlib import Path
import logging
from typing import List, Dict
import time
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "scraped_images"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FoodImageCollector:
    def __init__(
        self,
        output_dir: str = str(DATA_DIR),
        items_per_category: int = 25
    ):
        """
        Initialize the food image collector
        
        Args:
            output_dir: Directory to save images
            items_per_category: Number of images to collect per food category
        """
        self.output_dir = Path(output_dir)
        self.items_per_category = items_per_category
        self.food_categories = {
            "milk_carton": "french milk carton package supermarket",
            "eggs": "french eggs carton supermarket",
            "beef": "raw beef meat french supermarket",
            "chicken": "raw chicken meat french supermarket",
            "fish": "fresh fish french supermarket"
        }

    def setup_directories(self) -> None:
        """Create output directories for each category"""
        for category in self.food_categories.keys():
            category_dir = self.output_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            # Clean existing files if any
            if any(category_dir.iterdir()):
                shutil.rmtree(category_dir)
                category_dir.mkdir(parents=True)

    def collect_images(self, category: str, search_term: str) -> int:
        """
        Collect images for a specific category
        
        Args:
            category: Food category name
            search_term: Search term for the crawler
            
        Returns:
            int: Number of successfully downloaded images
        """
        output_dir = str(self.output_dir / category)
        
        crawler = BingImageCrawler(
            downloader_threads=4,
            storage={'root_dir': output_dir}
        )
        
        try:
            crawler.crawl(
                keyword=search_term,
                max_num=self.items_per_category * 2,  # Request more to ensure we get enough
                min_size=(200,200)
            )
        except Exception as e:
            logger.error(f"Error crawling {category}: {str(e)}")
            return 0
        
        downloaded_files = list(Path(output_dir).glob('*.jpg'))
        return len(downloaded_files)

    def cleanup_extra_images(self) -> Dict[str, int]:
        """
        Clean up extra images after all categories are collected
        
        Returns:
            Dict[str, int]: Number of remaining images per category
        """
        cleanup_stats = {}
        
        for category in self.food_categories.keys():
            category_dir = self.output_dir / category
            files = list(sorted(category_dir.glob('*.jpg')))
            
            if len(files) > self.items_per_category:
                # Remove extra images
                for file in files[self.items_per_category:]:
                    file.unlink()
                logger.info(f"Removed {len(files) - self.items_per_category} extra images from {category}")
            
            cleanup_stats[category] = min(len(files), self.items_per_category)
        
        return cleanup_stats

    def ensure_image_count(self, category: str, search_term: str) -> bool:
        """
        Ensure we get at least the required number of images
        
        Args:
            category: Food category name
            search_term: Search term for the crawler
            
        Returns:
            bool: True if successful, False otherwise
        """
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            count = self.collect_images(category, search_term)
            
            if count >= self.items_per_category:
                logger.info(f"Successfully collected {count} images for {category}")
                return True
                
            logger.warning(f"Attempt {attempt + 1}: Got only {count} images for {category}")
            attempt += 1
            time.sleep(2)
            
        logger.error(f"Failed to collect enough images for {category} after {max_attempts} attempts")
        return False

    def run(self) -> Dict[str, Dict]:
        """
        Run the image collection process for all categories
        
        Returns:
            Dict[str, Dict]: Results including collection success and final image counts
        """
        self.setup_directories()
        collection_results = {}
        
        # First collect all images
        for category, search_term in self.food_categories.items():
            logger.info(f"Collecting images for {category}...")
            collection_results[category] = {
                "success": self.ensure_image_count(category, search_term)
            }
        
        # Then cleanup extra images
        logger.info("Cleaning up extra images...")
        final_counts = self.cleanup_extra_images()
        
        # Combine results
        for category in self.food_categories.keys():
            collection_results[category]["final_count"] = final_counts[category]
        
        return collection_results

def main():
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Data directory: {DATA_DIR}")
    collector = FoodImageCollector()
    results = collector.run()
    
    print("\nCollection Results:")
    for category, result in results.items():
        status = "Success" if result["success"] else "Failed"
        print(f"{category}: {status} (Final count: {result['final_count']} images)")

if __name__ == "__main__":
    main()