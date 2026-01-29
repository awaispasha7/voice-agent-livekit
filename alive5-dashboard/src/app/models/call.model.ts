/**
 * HITL Voice Handoff - Call Models
 */

export interface IncomingCall {
  thread_id: string;
  crm_id: string;
  room_name: string;
  caller_phone: string;
  queue: string;
  timestamp: number;
  context: string;
}

export interface CallSession {
  room_name: string;
  thread_id: string;
  crm_id: string;
  channel_id: string;
  caller_phone: string;
  queue: string;
  agent_id?: string;
  agent_name?: string;
  started_at: Date;
  ended_at?: Date;
  duration?: number;
}

export interface HumanAgent {
  agent_id: string;
  agent_name: string;
  status: 'available' | 'busy' | 'offline';
  current_call?: string;
}

