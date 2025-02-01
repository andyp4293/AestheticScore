import { Component, ViewChild, ElementRef, AfterViewInit, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

declare var google: any;

interface BusinessInfo { // define the type of the business info  
  name: string;
  rating: number;
  totalRatings: number;
  placeId: string;
}

// decorators in Angular are function that modify the behavior of a class
// @Component is a decorator that tells Angular that this class is a component 
@Component({
  selector: 'app-search', // this names the component's html tag ie <app-search></app-search>
  standalone: false, // this tells Angular that this component is not standalone and needs to be included in a module
  templateUrl: './search.component.html', // this is the html file for the component
  styleUrls: ['./search.component.css'] // this is the css file for the component 
})
export class SearchComponent implements AfterViewInit, OnInit {
  @ViewChild('searchInput') searchInput!: ElementRef; // this is a reference to the search input element in the html file
  searchQuery: string = ''; // this is the query string for the search input
  selectedBusiness: BusinessInfo | null = null; // this is the business info for the selected business

  ngAfterViewInit() {  // this runs after you can see the component
    this.initAutocomplete();
  }

  ngOnInit() { // this runs before you can anything on the screen/component
    this.initAutocomplete();
  }

  private initAutocomplete() { // initializes the autocomplete feature for the search input
    if (this.searchInput) { // if there is any input so far
      const autocomplete = new google.maps.places.Autocomplete( // creates a new autocomplete object
        this.searchInput.nativeElement,
        { 
          types: ['establishment'], // this restricts autocomplete to businesses only 
          fields: ['name', 'place_id', 'rating', 'user_ratings_total', 'reviews'] // Explicitly request these fields
        }
      );

      autocomplete.addListener('place_changed', () => {  // this is an event listener, that runs when the user clicks on a business in the autocomplete dropdown
        const place = autocomplete.getPlace(); // this gets the place object from the autocomplete object
        console.log('Place details:', place); // Debug log
        
        if (place && place.place_id) { // if place is not null 
          this.selectedBusiness = { // sets selectedbusiness to the place object
            name: place.name || '',
            rating: Number(place.rating) || 0,
            totalRatings: Number(place.user_ratings_total) || 0,
            placeId: place.place_id
          };
          
          console.log('Selected business:', this.selectedBusiness); // Debug log
          
          this.searchQuery = place.name || ''; // sets to the search query to the name of the business
          this.analyzeBusiness(place.place_id); // calls the analyzeBusiness function with the place id
        }
      });
    }
  }

  analyzeBusiness(placeId?: string) {
    if (this.searchQuery.trim()) {
      console.log('Analyzing business:', this.searchQuery, 'Place ID:', placeId); 
      
    }
  }
}
