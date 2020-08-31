# test_html.py
from bs4 import BeautifulSoup
import requests

class TestWebpage:
    def get_soup(self):
            source = requests.get("http://0.0.0.0:8000")
            soup = BeautifulSoup(source.content, 'html.parser')
            return soup

    
    
    def test_textarea(self):
        soup = self.get_soup()
        #print("here I")
        assert soup.find_all('textarea')
    def test_button(self):
        soup = self.get_soup()
        #print("here I")
        assert soup.find_all('input',value='predict') 
        
    
    
        

# t=TestWebpage()
# t.test_textarea()
# t.test_button()    
# t.test_result()
    