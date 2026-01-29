// Production environment configuration
// Environment variables are injected at build time by Vercel
// @ts-ignore - process.env is available in Node.js build environment
const BACKEND_URL = typeof process !== 'undefined' && process.env ? process.env['BACKEND_URL'] : '';
// @ts-ignore
const A5_SOCKET_URL = typeof process !== 'undefined' && process.env ? process.env['A5_SOCKET_URL'] : '';
// @ts-ignore
const A5_API_KEY = typeof process !== 'undefined' && process.env ? process.env['A5_API_KEY'] : '';
// @ts-ignore
const LIVEKIT_URL = typeof process !== 'undefined' && process.env ? process.env['LIVEKIT_URL'] : '';

export const environment = {
  production: true,
  backendUrl: BACKEND_URL || 'https://18.210.238.67.nip.io',
  alive5SocketUrl: A5_SOCKET_URL || 'wss://api-stage.alive5.com',
  alive5ApiKey: A5_API_KEY || '7954047b-29d7-4098-aeca-9c309ab905da',
  livekitUrl: LIVEKIT_URL || 'ws://18.210.238.67:7880'
};

