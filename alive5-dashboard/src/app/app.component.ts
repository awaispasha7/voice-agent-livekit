import { Component, OnInit, OnDestroy } from '@angular/core';
import { Alive5SocketService } from './services/alive5-socket.service';
import { AgentProfileService, AgentProfile } from './services/agent-profile.service';
import { environment } from '../environments/environment';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit, OnDestroy {
  title = 'Alive5 HITL Dashboard';
  agentProfile: AgentProfile | null = null;

  constructor(
    private alive5Socket: Alive5SocketService,
    private agentProfileService: AgentProfileService
  ) {}

  ngOnInit(): void {
    console.log('[App] Initializing dashboard...');
    
    // Prompt for agent name if not set
    if (!this.agentProfileService.hasProfile()) {
      this.promptForAgentName();
    }
    
    // Load and subscribe to profile changes
    this.agentProfile = this.agentProfileService.getProfile();
    this.agentProfileService.profile$.subscribe(profile => {
      this.agentProfile = profile;
    });
    
    // Get agent ID from profile or generate a temporary one
    const profile = this.agentProfileService.getProfile();
    const agentId = profile?.agentId || `dashboard_${Date.now()}`;
    
    // Connect to Alive5 Socket.IO
    try {
      this.alive5Socket.connect(
        environment.alive5ApiKey,
        agentId,
        environment.alive5SocketUrl
      );
      console.log('[App] Socket connection initiated');
    } catch (error) {
      console.error('[App] Failed to initialize socket:', error);
    }
  }

  ngOnDestroy(): void {
    console.log('[App] Disconnecting socket...');
    this.alive5Socket.disconnect();
  }

  /**
   * Prompt user to enter their name
   */
  private promptForAgentName(): void {
    const name = prompt('Welcome to Alive5 HITL Dashboard!\n\nPlease enter your name:');
    
    if (name && name.trim()) {
      this.agentProfileService.setProfile(name.trim());
      console.log('[App] Agent profile created:', name);
    } else {
      // If no name provided, prompt again after a short delay
      setTimeout(() => {
        if (!this.agentProfileService.hasProfile()) {
          this.promptForAgentName();
        }
      }, 1000);
    }
  }

  /**
   * Allow user to edit their name
   */
  editAgentName(): void {
    const currentName = this.agentProfile?.agentName || '';
    const newName = prompt('Enter your name:', currentName);
    
    if (newName && newName.trim() && newName.trim() !== currentName) {
      this.agentProfileService.updateName(newName.trim());
      console.log('[App] Agent name updated:', newName);
    }
  }
}

