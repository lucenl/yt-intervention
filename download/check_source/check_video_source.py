from ytdriver import YTDriver, Video
import time


def main():
    # Set your local Chrome profile path.
    profile_dir = "/Users/lilucen/Library/Application Support/Google/Chrome"

    # Create a YTDriver instance with your profile.
    driver = YTDriver(verbose=True, use_virtual_display=False, headless=False, version_main=133)

    # Navigate to YouTube.
    driver.driver.get("https://www.youtube.com")
    time.sleep(10)  # Wait for the page to load.

    # Retrieve homepage recommendations.
    homepage_recs = driver.get_homepage_recommendations(scroll_times=0)

    # Save screenshot of the homepage.
    driver.save_screenshot("homepage.png")

    for video in homepage_recs:
        print(video.videoId)

    # Save the recommendations to a txt file.
    with open("homepage_recommendations.txt", "w") as f:
        for video in homepage_recs:
            f.write(video.videoId + "\n")

    # When done, close the driver.
    driver.close()


if __name__ == '__main__':
    main()
