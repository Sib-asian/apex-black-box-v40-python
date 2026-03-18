import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class StreamlitApp:
    def __init__(self):
        self._apex_last_req_id = None

    def start(self):
        logging.info("Application started.")

    def handle_request(self, request_data):
        logging.info(f"Received request: {request_data}")
        try:
            # Robust parsing of component_value to accept wrapped {value:{...}}
            component_value = request_data.get('component_value')
            if isinstance(component_value, dict) and 'value' in component_value:
                component_value = component_value['value']

            # Process component_value...
            logging.info(f"Processed component value: {component_value}")

            # Ensure final action sets _apex_last_req_id
            self._apex_last_req_id = request_data.get('req_id')
            logging.info(f"Set last request ID: {self._apex_last_req_id}")

            # Trigger rerun to flush any pending render
            self.trigger_rerun()

        except Exception as e:
            logging.error(f"Error processing request: {e}")

    def trigger_rerun(self):
        logging.info("Triggering rerun to flush pending render")
        # Code to trigger rerun

# Example usage
if __name__ == '__main__':
    app = StreamlitApp()
    app.start()