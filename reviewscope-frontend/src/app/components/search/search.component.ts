import { Component, ViewChild, ElementRef, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

declare var google: any;

@Component({
  selector: 'app-search',
  standalone: false,
  templateUrl: './search.component.html',
  styleUrls: ['./search.component.css']
})
export class SearchComponent implements AfterViewInit {
  @ViewChild('searchInput') searchInput!: ElementRef;
  searchQuery: string = '';
  private autocomplete: any;

  ngAfterViewInit() {
    // Initialize Google Places Autocomplete
    this.autocomplete = new google.maps.places.Autocomplete(
      this.searchInput.nativeElement,
      {
        types: ['establishment'],  // This restricts to businesses only
        fields: ['place_id', 'name', 'formatted_address']
      }
    );

    // Handle place selection
    this.autocomplete.addListener('place_changed', () => {
      const place = this.autocomplete.getPlace();
      if (place.place_id) {
        this.searchQuery = place.name;
        this.analyzeBusiness(place.place_id);
      }
    });
  }

  analyzeBusiness(placeId?: string) {
    if (this.searchQuery.trim()) {
      console.log('Analyzing business:', this.searchQuery, 'Place ID:', placeId);
      // Add your analysis logic here
    }
  }
}
