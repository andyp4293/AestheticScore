import { Component, EventEmitter, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-search',
  standalone: false,
  templateUrl: './search.component.html',
  styleUrls: ['./search.component.css']
})
export class SearchComponent {
  @Output() searchResults = new EventEmitter<any>();
  searchQuery: string = '';

  searchBusiness() {
    // Placeholder for now
    console.log('Searching for:', this.searchQuery);
  }
}
