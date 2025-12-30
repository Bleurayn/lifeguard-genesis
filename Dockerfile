# docker-compose.yml
services:
  lifeguard:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data  # persists lifeguard_public.db

volumes:
  data:
