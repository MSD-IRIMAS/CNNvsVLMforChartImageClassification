# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 21:13:55 2019

@author: Bruno
"""

import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import urllib
import glob
from PIL import Image


def convert_all_img_in_jpg(download_path, words_to_search):
    for word in words_to_search:
        wordFolder = glob.glob(download_path + word.replace(" ", "_") + "/*.png")
        print('#######################################################')
        print('Converting images in folder ' + word.replace(" ", "_") + ' to jpg')
        for img in wordFolder:
            im = Image.open(img)
            rgb_im = im.convert("RGB")
            rgb_im.save(img.replace("png", "jpg"), "jpeg", quality=100)
            os.remove(img)
        print('Conversion en jpg terminée !')


def search_and_save(download_path, word, imgs_nb, firstImgPosition):
    scrolls_nb = 10
    print("Search: "+ word +" | Images nb: "+ str(imgs_nb)+" | first image position: "+str(firstImgPosition))
    
    # Creates the folder to save the images
    if not os.path.exists(download_path + word.replace(" ", "_")):
        os.makedirs(download_path + word.replace(" ", "_"))
    
    # Launch Chrome
    service = Service(executable_path='chromedriver.exe') # Indicates the driver location
    driver = webdriver.Chrome(service=service) # WebDriver used is Chrome
    
    # Launch the search
    url = "https://www.google.com/search?q=" + word + "&source=lnms&tbm=isch"
    driver.get(url)
    headers = {}
    headers['User-Agent'] = "Mozilla/5.0 (compatible; Googlebot/2.1; +https://www.google.com/bot.html)"
    
    # Click on the "Reject all" button
    WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable(
            (
                By.XPATH,
                "//button[contains(@aria-label, 'Reject all')]"
            )
        )
    ).click()
    
    for _ in range(scrolls_nb):
        for __ in range(10):
            driver.execute_script("window.scrollBy(0, 1000000)")
            time.sleep(2)
        # to load next images
        time.sleep(2)
        try:
            driver.find_element(By.XPATH, '//*[@class="mye4qd"]').click()
            time.sleep(2)
        except Exception as e:
            print("Fewer images found: " + str(e))
            break
    
    imagesDiv = driver.find_element(By.ID, "islrg")
    imgs = imagesDiv.find_elements(By.TAG_NAME, "img")
    print("Images found: "+ str(len(imgs)) + "\n")
    
    img_count = 0
    downloaded_img_count = 0
    img_skip = 0
    
    for img in imgs:
        img_count += 1
        if img_skip < firstImgPosition:
            img_skip += 1
        else:
            img_url = img.get_attribute('src')
            img_type = "png"
            try:
                if img_url != None:
                    img_url = str(img_url)
                    # Download and save the image
                    req = urllib.request.Request(img_url, headers=headers)
                    raw_img = urllib.request.urlopen(req).read()
                    chartClass = word.replace(" ", "_")
                    imgPath = download_path+chartClass+"/"+chartClass+"_"+str(img_skip+downloaded_img_count)+"."+img_type
                    f = open(imgPath, "wb")
                    f.write(raw_img)
                    f.close
                    downloaded_img_count += 1
                    if downloaded_img_count == 1 or downloaded_img_count % 10 == 0:
                        print('Image n° ' + str(downloaded_img_count) + ' downloaded!')
                else:
                    img_skip += 1
                    print('Download of image n° ' + str(img_count) + ' impossible because its url = ' + str(img_url))
            except Exception as e:
                print('Download failed: ' + str(e))
            finally:
                if downloaded_img_count >= imgs_nb:
                    break
    
    print("Total skipped: "+str(img_skip)+"; Total downloaded: "+ str(downloaded_img_count)+ "/"+ str(img_count))
    driver.quit()
    

if __name__ == "__main__":
    download_path = "chartClasses/"
    
    words_to_search = ['area chart', 'bar chart','barcode plot', 'boxplot',
                       'bubble chart', 'column chart', 'diverging bar chart',
                       'diverging stacked bar chart', 'donut chart', 'dot strip plot',
                       'heatmap', 'line chart', 'line column chart',
                       'lollipop chart', 'ordered bar chart', 'ordered column chart',
                       'paired bar chart', 'paired column chart', 'pie chart',
                       'population pyramid', 'proportional stacked bar chart',
                       'scatter plot', 'spine chart', 'stacked column chart',
                       'violin plot']

    images_nb = 500
    first_img_position = 0
    
    i = 0
    while i < len(words_to_search):
        print('########################################################')
        print('Search term: ' + words_to_search[i])
        print('########################################################')
        if images_nb > 0:
            search_and_save(download_path, words_to_search[i], images_nb,
                            first_img_position)
        i += 1
    
    convert_all_img_in_jpg(download_path, words_to_search)
    
