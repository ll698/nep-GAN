# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractor import LinkExtractor
from scrapy.spiders import Rule, CrawlSpider
from nep_scrape.items import NepScrapeItem


class CpphotofinderSpider(scrapy.Spider):
    name = 'cpphotofinder'
    start_urls = ['http://cpphotofinder.com/search.php?genus=Nepenthes']

    def parse(self, response):
        # page = response.url.split("/")[-2]
        # print(page)
        # filename = 'quotes-%s.html' % page
        # print(filename)
        # with open(filename, 'wb') as f:
        #     f.write(response.body)
        # self.log('Saved file %s' % filename)

        # items = []
        # # Only extract canonicalized and unique links (with respect to the current page)
        # links = LinkExtractor().extract_links(response)
        # # Now go through all the found links
        # for link in links:
        #     # Check whether the domain of the URL of the link is allowed; so whether it is in one of the allowed domains
        #     print(link.url)
        #     yield scrapy.Request(link.url, self.parse_page)
        #     print("hello")
        #     break
        # return items
       count = 0
       for href in response.xpath("//a"):
            count+=1
            url = href.xpath("@href").extract_first()
            if "http" in url and "jpg" in url:
                yield scrapy.Request(url,
                self.parse_img)
        # Return all the found items


    def parse_img(self, response):
        try:
            img = response.xpath("//img")
            print("pass")
        except:
            print("fail")
       

        # print(img)
        # imageURL = img.extract_first()
        # print(imageURL)
        # yield NepScrapeItem(title=title, pubDate=pub, file_urls=[imageURL])