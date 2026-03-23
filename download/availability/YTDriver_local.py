import undetected_chromedriver as uc
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import Chrome, ChromeOptions, ChromeService, Firefox, FirefoxOptions
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException
from time import sleep
from helpers import Video, VideoUnavailableException, time2seconds
from pyvirtualdisplay import Display
import os

class YTDriver:

    AD_CLASSNAMES = [
        'ytp-preview-ad',
        'ytp-ad-preview-container'
    ]

    def __init__(self, profile_dir=None, use_virtual_display=False, headless=False, verbose=False, version_main=None,):
        """
        Initializes the webdriver and virtual display

        ### Arguments:
        - `profile_dir`: Specify a directory to save the browser profile so it can be loaded later. Set to `None` to not save the profile.
        - `use_virtual_display`: Set to `True` to launch a virtual display using `pyvirtualdisplay`.
        - `headless`: Set to `True` to run the browser in headless mode.
        - `verbose`: Set to `True` to enable logging messages.
        """

        self.verbose = verbose

        if use_virtual_display:
            self.__log("Starting virtual display")
            display = Display(size=(1920,1080))
            display.start()

        self.__init_chrome(profile_dir, headless, version_main)
        self.driver.set_page_load_timeout(30)

    def close(self):
        """
        Close the underlying webdriver.
        """
        self.driver.close()

    def clear_history(self):
        """
        Clear the user watch history.
        """
        self.driver.get('https://www.youtube.com/feed/history')
        try:
            clear_button = WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.XPATH, '//button[.="Clear all watch history"]'))
            )
            clear_button.click()
        except: pass

        try:
            confirm_button = WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.XPATH, '//button[.="Clear watch history"]'))
            )
            confirm_button.click()
        except: pass

    def login(self, username, password):
        # go to the homepage first
        self.get_homepage_recommendations()

        # click on signin
        sign_in = self.driver.find_elements(By.XPATH, '//a[@aria-label="Sign in"]')
        
        if len(sign_in) == 0:
            # no sign in button => already signed in
            return

        # click sign in button
        sign_in[0].click()

        # type in email
        self.driver.find_element(By.XPATH, '//input[@type="email"]').send_keys(username)
        sleep(3)

        # click on next
        self.driver.find_element(By.XPATH, '//span[text()="Next"]').click()
        sleep(3)

        # type in password
        self.driver.find_element(By.XPATH, '//input[@type="password"]').send_keys(password)
        sleep(3)

        # click on next
        self.driver.find_element(By.XPATH, '//span[text()="Next"]').click()
        sleep(3)

        # click on not now if asking for address
        try: self.driver.find_element(By.XPATH, '//span[text()="Not now"]').click()
        except: pass

    def get_homepage_recommendations(self, scroll_times=0) -> list[Video]:
        """
        Collect videos from the YouTube homepage.

        ### Arguments:
        - `scroll_times`: Number of times to scroll the homepage.

        ### Returns:
        - List of videos of type `ytdriver.helpers.Video`.

        """
        # try to find the youtube icon
        try:
            self.__log('Clicking homepage icon')
            self.driver.find_element(By.ID, 'logo-icon').click()
        except:
            self.__log('Getting homepage via URL')
            self.driver.get('https://www.youtube.com')

        # wait for page to load
        sleep(2)

        # scroll page to load more results
        for _ in range(scroll_times):
            self.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.PAGE_DOWN)
            sleep(0.2)
            

        # collect video-like tags from homepage
        videos = self.driver.find_elements(By.XPATH, '//div[@id="contents"]/ytd-rich-item-renderer')

        # identify actual videos from tags
        homepage = []
        for video in videos:
            a = video.find_elements(By.TAG_NAME, 'a')[0]
            href = a.get_attribute('href')
            if href is not None and href.startswith('https://www.youtube.com/watch?'):
                homepage.append(Video(a, href))

        return homepage

    def get_upnext_recommendations(self, topn=5) -> list[Video]:

        """
        Collect up-next recommendations for the currently playing video.

        ### Arguments:
        - `topn`: Number of recommendations to return.

        ### Returns:
        - List of videos of type `ytdriver.helpers.Video`.
        
        """
        # wait for page to load
        sleep(2)

        # wait for recommendations
        elems = WebDriverWait(self.driver, 30).until(
            EC.presence_of_all_elements_located((By.XPATH, '//ytd-compact-video-renderer|//ytd-rich-item-renderer'))
        )

        # recommended videos array
        elems = [el for el in elems if el.is_displayed()]
        return [Video(elem, elem.find_element(By.TAG_NAME, 'a').get_attribute('href')) for elem in elems[:topn]]

    def search_videos(self, query, scroll_times=0) -> list[Video]:
        """
        Search for videos.

        ### Arguments:
        - `query` (`str`): Search query.

        ### Returns:
        - List of videos of type `ytdriver.helpers.Video`.
        
        """

        # load video search results
        self.driver.get('https://www.youtube.com/results?search_query=%s' % query)

        # wait for page to load
        sleep(2)

        # scroll page to load more results
        for _ in range(scroll_times):
            self.driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.PAGE_DOWN)
            sleep(0.2)

        # collect video-like tags from homepage
        videos = self.driver.find_elements(By.XPATH, '//div[@id="contents"]/ytd-video-renderer')
        
        # identify actual videos from tags
        results = []
        for video in videos:
            a = video.find_elements(By.TAG_NAME, 'a')[0]
            href = a.get_attribute('href')
            if href is not None and href.startswith('https://www.youtube.com/watch?'):
                results.append(Video(a, href))
        return results


    def play(self, video, duration=5):
        """
        Play a video for a set duration. Returns when that duration passes.

        ### Arguments:
        - `video` (`str`|`ytdriver.helpers.Video`): Video object or URL to play.
        - `duration` (`int`): How long to play the video.
        
        """
        try:
            self.__click_video(video)
            self.__check_video_availability()
            self.__click_play_button()
            self.__handle_ads()
            self.__clear_prompts()
            sleep(duration)
        except WebDriverException as e:
            self.__log(e)

    def save_screenshot(self, filename):
        """
        Save a screenshot of the current browser window.

        ### Arguments:
        - `filename`: Filename to save image as.
        """
        return self.driver.save_screenshot(filename)

    ## Helpers
    def __log(self, message):
        if self.verbose:
            print(message)

    def __click_video(self, video):
        if type(video) == Video:
            try:
                # try to click the element using selenium
                self.__log("Clicking element via Selenium...")
                video.elem.click()
                return
            except Exception as e:
                try:
                    # try to click the element using javascript
                    self.__log("Failed. Clicking via Javascript...")
                    self.driver.execute_script('arguments[0].click()', video.elem)
                except:
                    # js click failed, just open the video url
                    self.__log("Failed. Loading video URL...")
                    self.driver.get(video.url)
        elif type(video) == str:
            self.driver.get(video)
        else:
            raise ValueError('Unsupported video parameter!')

    def __check_video_availability(self):
        try:
            WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.XPATH, '//*[@id="container"]/h1'))
            )
        except WebDriverException:
            raise VideoUnavailableException()

    def __click_play_button(self):
        try:
            playBtn = self.driver.find_elements(By.CLASS_NAME, 'ytp-play-button')
            if 'Play' in playBtn[0].get_attribute('title'):
                playBtn[0].click()
        except:
            pass

    def __handle_ads(self):
        # handle multiple ads
        while True:
            sleep(1)

            # check if ad is being shown
            xpath = '//div[%s]' % (' or '.join(["@class='%s'" % i for i in YTDriver.AD_CLASSNAMES]))
            ad_elems = self.driver.find_elements(By.XPATH, xpath)
            if len(ad_elems) == 0:
                self.__log('Ad not detected')
                # ad is not shown, return
                return

            self.__log('Ad detected')
            
            # an ad is being shown
            # grab preview text to determine ad type
            preview = ad_elems[0]
            text = preview.text.replace('\n', ' ').strip()
            wait = 0
            if 'after ad' in text or 'plays soon' in text:
                # unskippable ad, grab ad length
                length = self.driver.find_elements(By.CLASS_NAME, 'ytp-ad-duration-remaining')[0].text
                wait = time2seconds(length)
                self.__log('Unskippable ad. Waiting %d seconds...' % wait)
            elif 'begin in' in text or 'end in' in text:
                # short ad
                wait = int(text.split()[-1])
                self.__log('Short ad. Waiting for %d seconds...' % wait)
            else:
                # skippable ad, grab time before skippable
                if text == '':
                    text = '10'
                wait = int(text)
                self.__log('Skippable ad. Skipping after %d seconds...' % wait)

            # wait for ad to finish
            sleep(wait)

            # click skip button if available
            skip = self.driver.find_elements(By.CLASS_NAME, 'ytp-skip-ad-button')
            if len(skip) > 0:
                skip[0].click()

    def __clear_prompts(self):
        try:
            sleep(1)
            self.driver.find_element(By.XPATH, '/html/body/ytd-app/ytd-popup-container/tp-yt-iron-dropdown/div/yt-tooltip-renderer/div[2]/div[1]/yt-button-renderer/a/tp-yt-paper-button/yt-formatted-string').click()
        except:
            pass
    
    def __init_chrome(self, profile_dir, headless, version_main):
        self.driver = uc.Chrome(user_data_dir=profile_dir, headless=headless, use_subprocess=False, version_main=version_main)



