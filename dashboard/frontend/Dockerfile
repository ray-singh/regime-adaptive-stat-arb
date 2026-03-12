FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY dashboard/frontend/package*.json ./
RUN npm ci --silent

# Copy source
COPY dashboard/frontend/ ./

# Expose Vite dev server port and run in host mode so it's reachable
EXPOSE 5173

# Use host 0.0.0.0 so the dev server is accessible from outside the container
CMD ["sh", "-c", "npm run dev -- --host 0.0.0.0"]
